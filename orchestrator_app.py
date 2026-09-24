import os
import re

import streamlit as st

from orchestrator import (
    AGENTS,
    AGENTS_BY_ID,
    get_agent_dataset,
    route_question,
)
from agent1.main import resolve_follow_up

# Run with:
# python -m streamlit run orchestrator_app.py

st.set_page_config(
    page_title="Multi-Agent Text-to-SQL",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Multi-Agent Text-to-SQL")
st.write(
    "Ask a question and it will be routed automatically to whichever "
    "registered agent's dataset can answer it."
)

st.sidebar.subheader("Registered agents")
for _agent in AGENTS:
    st.sidebar.markdown(f"**{_agent['name']}**")
    st.sidebar.caption(_agent["core"].CSV_PATH)

reload_clicked = st.sidebar.button("Reload all datasets")


# ============================================================
# CREATE / LOAD DATABASES (cached across reruns/questions)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_all_datasets(agent_mtimes):
    """agent_mtimes is a tuple of (agent_id, mtime) pairs, passed in purely
    so Streamlit's cache key changes whenever ANY underlying CSV changes on
    disk - the actual value isn't used below, get_agent_dataset() re-derives
    everything from each agent's own CSV_PATH."""
    return {agent["id"]: get_agent_dataset(agent) for agent in AGENTS}


def _current_mtimes():
    mtimes = []
    for agent in AGENTS:
        try:
            mtimes.append((agent["id"], os.path.getmtime(agent["core"].CSV_PATH)))
        except OSError as error:
            st.error(f"Could not find the CSV file for {agent['name']}: {error}")
            st.stop()
    return tuple(mtimes)


if reload_clicked:
    get_all_datasets.clear()

try:
    with st.spinner("Preparing datasets and schemas for all agents..."):
        datasets = get_all_datasets(_current_mtimes())
except Exception as error:
    st.error(f"Could not initialize one or more datasets: {error}")
    st.stop()


# Clear conversation automatically if any agent's dataset changes
fingerprint = tuple(sorted((aid, d["fingerprint"]) for aid, d in datasets.items()))
if st.session_state.get("dataset_fingerprint") != fingerprint:
    st.session_state.messages = []
    st.session_state.dataset_fingerprint = fingerprint


# ============================================================
# SCHEMA DOWNLOADS (one per agent)
# ============================================================

for agent in AGENTS:
    dataset = datasets[agent["id"]]
    with open(dataset["schema_path"], encoding="utf-8") as schema_file:
        st.sidebar.download_button(
            f"Download {agent['id']} schema.md",
            schema_file.read(),
            file_name=f"{agent['id']}_schema.md",
            mime="text/markdown",
            key=f"schema_{agent['id']}",
        )


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

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
# CHART RENDERING (identical behavior to each single-agent app)
# ============================================================

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

        # Which agent handled this turn, and why - shown only on assistant
        # replies, since the router only runs once the question is asked.
        if message.get("agent_name"):
            caption = f"Routed to: {message['agent_name']}"
            if message.get("routing_reason"):
                caption += f"  \u2014 {message['routing_reason']}"
            st.caption(caption)

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
# Routing happens first (against the whole registry), then everything below
# runs through THAT agent's own functions - so each agent's own MODEL_NAME,
# chart specs, and behavior are respected exactly as in its single-agent app.

def answer_question(standalone, history):
    with st.spinner("Routing to the right agent..."):
        agent_id, reasoning = route_question(standalone, AGENTS, history)

    agent = AGENTS_BY_ID[agent_id]
    core = agent["core"]
    dataset = datasets[agent_id]

    message = {"role": "assistant", "agent_name": agent["name"], "routing_reason": reasoning}

    try:
        # 1. Explicit chart-only requests ("show me a chart of...") with no
        # narration wanted: Python draws them directly, no LLM narration call.
        if core.wants_charts(standalone):
            with st.spinner(f"Building charts from {agent['name']}..."):
                charts, label = core.build_charts(standalone, dataset)
            message["charts"] = charts
            message["content"] = (
                f"Here is a visual summary of {label} ({agent['name']})."
                if charts else
                "I couldn't build charts for that scope. Check that it matches data in the file."
            )

        # 2. Broad analysis: several queries, planned and narrated together.
        elif core.classify_question(standalone):
            with st.spinner(f"Planning and running multiple SQL queries on {agent['name']}..."):
                result = core.multi_query_analysis(standalone, dataset)
            message["content"] = result["text"]
            message["charts"] = result["charts"]

        # 3. Specific lookup: one SQL query
        else:
            with st.spinner(f"Checking {agent['name']}'s data..."):
                sql = core.generate_sql(
                    standalone,
                    schema_path=dataset["schema_path"],
                    db_path=dataset["db_path"],
                )
                message["sql"] = sql

                if not core.validate_sql(sql):
                    raise ValueError(
                        "The generated SQL was rejected by the safety validator."
                    )

                result_df = core.execute_sql(sql, db_path=dataset["db_path"])
                message["result"] = result_df
                message["content"] = core.generate_answer(standalone, sql, result_df)

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

if prompt := st.chat_input("Ask a question — it'll be routed to the right agent"):

    question = prompt.strip()

    if question:
        # History BEFORE adding the new message - used both to resolve a
        # follow-up into a standalone question and, unchanged, to route it.
        history = list(st.session_state.messages)

        # Self-contained questions come back unchanged; follow-ups such as
        # "Now only Premium" become full standalone questions. This runs
        # BEFORE routing so the router never sees a bare fragment.
        with st.spinner("Understanding your question..."):
            standalone = resolve_follow_up(question, history)

        user_message = {"role": "user", "content": question}
        if standalone != question:
            user_message["content"] = f"{question} (interpreted as: {standalone})"

        st.session_state.messages.append(user_message)
        display_message(user_message, msg_index=len(st.session_state.messages) - 1)

        assistant_message = answer_question(standalone, history)

        st.session_state.messages.append(assistant_message)
        display_message(assistant_message, msg_index=len(st.session_state.messages) - 1)