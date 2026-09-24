import os
import re


import streamlit as st

from main import (
    create_database,
    generate_sql,
    validate_sql,
    execute_sql,
    generate_answer,
    classify_question,
    multi_query_analysis,
    resolve_follow_up,
    wants_charts,
    build_charts,
    CSV_PATH,
)

# Run with:
# python -m streamlit run app.py

st.set_page_config(
    page_title="Local Text-to-SQL",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Local Text-to-SQL")
st.write("Ask questions about your CSV dataset and get answers in plain English.")

csv_path = st.sidebar.text_input(
    "CSV file path",
    value=CSV_PATH,
)

st.sidebar.caption(
    "Use a UTF-8, comma-separated CSV with a header row. "
    "Change the path or replace the file, then rerun the app."
)

reload_clicked = st.sidebar.button("Reload dataset")


# ============================================================
# CREATE / LOAD DATABASE (cached across reruns/questions)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_dataset(path: str, file_mtime: float):
    return create_database(path)


try:
    file_mtime = os.path.getmtime(csv_path)
except OSError as error:
    st.error(f"Could not find the CSV file: {error}")
    st.stop()

if reload_clicked:
    get_dataset.clear()

try:
    with st.spinner("Preparing dataset and schema with Qwen..."):
        dataset = get_dataset(csv_path, file_mtime)

except Exception as error:
    st.error(f"Could not initialize the dataset: {error}")
    st.stop()


# Clear conversation automatically if the dataset changes
if st.session_state.get("dataset_fingerprint") != dataset["fingerprint"]:
    st.session_state.messages = []
    st.session_state.dataset_fingerprint = dataset["fingerprint"]


# ============================================================
# SCHEMA DOWNLOAD
# ============================================================

with open(dataset["schema_path"], encoding="utf-8") as schema_file:
    st.sidebar.download_button(
        "Download schema.md",
        schema_file.read(),
        file_name="schema.md",
        mime="text/markdown",
    )


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# CLEAR CONVERSATION
# ============================================================

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []


# ============================================================
# MARKDOWN / CURRENCY FIX
# ============================================================

def escape_markdown_math(text):
    """
    Streamlit's markdown renderer treats $...$ as LaTeX math,
    which can mangle plain currency amounts such as:

        $50,899,872.43

    Escape unescaped $ signs so they render as normal text.
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


# ============================================================
# CHART RENDERING
# ============================================================
# Each chart dict now carries: title, data (DataFrame with label/value
# columns), chart_type (the suggested default: 'bar' | 'pie' | 'line'), and
# available_types (what else the user could switch to). This renders the
# default type and offers a selector to switch, per chart, without re-running
# any SQL or LLM calls - it's the same underlying data either way.

_CHART_LABELS = {"bar": "Bar", "pie": "Pie", "line": "Line"}


def render_chart(chart, key_prefix):
    st.subheader(chart["title"])

    available = chart.get("available_types", [chart.get("chart_type", "bar")])
    default_type = chart.get("chart_type", available[0])

    if len(available) > 1:
        choice = st.radio(
            "Chart type",
            options=available,
            format_func=lambda t: _CHART_LABELS.get(t, t.title()),
            index=available.index(default_type) if default_type in available else 0,
            horizontal=True,
            key=f"{key_prefix}_{chart['title']}",
        )
    else:
        choice = default_type

    data = chart["data"]

    if choice == "pie":
        # st.bar_chart/st.line_chart have no native pie option; use a small
        # Plotly figure only for this case, matplotlib-free and dependency-light.
        try:
            import plotly.express as px
            fig = px.pie(data, names="label", values="value", title=None)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Pie charts require the `plotly` package; showing a bar chart instead.")
            st.bar_chart(data, x="label", y="value")
    elif choice == "line":
        st.line_chart(data, x="label", y="value")
    else:
        st.bar_chart(data, x="label", y="value")


# ============================================================
# DISPLAY A CHAT MESSAGE
# ============================================================

def display_message(message, msg_index=0):
    with st.chat_message(message["role"]):

        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(escape_markdown_math(message["content"]))

        # Charts: either explicitly requested ("show me a chart of...") or
        # attached automatically as the default visualization for any
        # analysis answer. Same rendering path either way.
        for i, chart in enumerate(message.get("charts", [])):
            render_chart(chart, key_prefix=f"msg{msg_index}_chart{i}")

        # SQL / debug details (single-query path)
        if message.get("sql"):
            with st.expander("Query details"):
                st.code(message["sql"], language="sql")

                if "result" in message:
                    st.dataframe(message["result"], use_container_width=True)

        # Raw error is shown for every path, not only when SQL exists
        if message.get("raw_error"):
            with st.expander("Error details"):
                st.code(message["raw_error"])


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for i, message in enumerate(st.session_state.messages):
    display_message(message, msg_index=i)


# ============================================================
# ANSWER ONE QUESTION
# ============================================================

def answer_question(standalone):
    """Route a standalone question and return the assistant message dict."""
    message = {"role": "assistant"}

    try:
        # 1. Explicit chart-only requests ("show me a chart of...") with no
        # narration wanted: Python draws them directly, no LLM narration call.
        if wants_charts(standalone):
            with st.spinner("Building charts..."):
                charts, label = build_charts(standalone, dataset)
            message["charts"] = charts
            message["content"] = (
                f"Here is a visual summary of {label}."
                if charts else
                "I couldn't build charts for that scope. Check that it matches data in the file."
            )

        # 2. Broad analysis: several queries, or the year-scoped analysis.
        # multi_query_analysis now always returns both the narrated text AND
        # a default set of the most relevant charts in one call, so no
        # separate build_charts() call is needed on this path.
        elif classify_question(standalone):
            with st.spinner("Planning and running multiple SQL queries..."):
                result = multi_query_analysis(standalone, dataset)
            message["content"] = result["text"]
            message["charts"] = result["charts"]

        # 3. Specific lookup: one SQL query
        else:
            with st.spinner("Checking the data with Qwen..."):
                sql = generate_sql(
                    standalone,
                    schema_path=dataset["schema_path"],
                    db_path=dataset["db_path"],
                )
                message["sql"] = sql

                if not validate_sql(sql):
                    raise ValueError(
                        "The generated SQL was rejected by the safety validator."
                    )

                result_df = execute_sql(sql, db_path=dataset["db_path"])
                message["result"] = result_df
                message["content"] = generate_answer(standalone, sql, result_df)

    except Exception as error:
        message.update(
            content=(
                "I couldn't complete that request. Try rephrasing the question, "
                "or open the error details below."
            ),
            raw_error=str(error),
            error=True,
        )

    return message


# ============================================================
# USER QUESTION
# ============================================================

if prompt := st.chat_input("Ask a question about this dataset"):

    question = prompt.strip()

    if question:
        # History BEFORE adding the new message
        history = list(st.session_state.messages)

        # Self-contained questions come back unchanged; follow-ups such as
        # "Now only Premium" become full standalone questions.
        with st.spinner("Understanding your question..."):
            standalone = resolve_follow_up(question, history)

        user_message = {"role": "user", "content": question}
        if standalone != question:
            user_message["content"] = f"{question} (interpreted as: {standalone})"

        st.session_state.messages.append(user_message)
        display_message(user_message, msg_index=len(st.session_state.messages) - 1)

        assistant_message = answer_question(standalone)

        st.session_state.messages.append(assistant_message)
        display_message(assistant_message, msg_index=len(st.session_state.messages) - 1)