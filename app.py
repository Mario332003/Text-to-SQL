import streamlit as st

from main import create_database, generate_sql, validate_sql, execute_sql, generate_answer, CSV_PATH

# Run with: python -m streamlit run app.py
st.set_page_config(page_title="Local Text-to-SQL", page_icon="🤖", layout="wide")


st.title("🤖 Local Text-to-SQL")
st.write("Ask questions about your CSV dataset and get answers in plain English.")
csv_path = st.sidebar.text_input("CSV file path", value=CSV_PATH)
st.sidebar.caption("Use a UTF-8, comma-separated CSV with a header row. Change the path or replace the file, then rerun the app.")
st.sidebar.button("Reload dataset")

try:
    with st.spinner("Preparing dataset and schema with Qwen..."):
        dataset = create_database(csv_path)
except Exception as error:
    st.error(f"Could not initialize the dataset: {error}")
    st.stop()

if st.session_state.get("dataset_fingerprint") != dataset["fingerprint"]:
    st.session_state.messages = []
    st.session_state.dataset_fingerprint = dataset["fingerprint"]

with open(dataset["schema_path"], encoding="utf-8") as schema_file:
    st.sidebar.download_button("Download schema.md", schema_file.read(),
                               file_name="schema.md", mime="text/markdown")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []


def display_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("sql"):
            with st.expander("Query details"):
                st.code(message["sql"], language="sql")
                if "result" in message:
                    st.dataframe(message["result"], use_container_width=True)


for message in st.session_state.messages:
    display_message(message)

if prompt := st.chat_input("Ask a question about this dataset"):
    question = prompt.strip()
    if question:
        history = list(st.session_state.messages)
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        display_message(user_message)
        assistant_message = {"role": "assistant"}
        try:
            with st.spinner("Checking the data with Qwen..."):
                sql = generate_sql(question, history=history, schema_path=dataset["schema_path"])
                assistant_message["sql"] = sql
                if not validate_sql(sql):
                    raise ValueError("The generated SQL was rejected by the safety validator.")
                result = execute_sql(sql, db_path=dataset["db_path"])
                assistant_message["result"] = result
                assistant_message["content"] = generate_answer(question, sql, result)
        except Exception as error:
            assistant_message.update(
                content=f"I couldn't answer this question: {error}", error=True
            )
        st.session_state.messages.append(assistant_message)
        display_message(assistant_message)
