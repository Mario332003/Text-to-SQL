import streamlit as st

from main import create_database, generate_sql, validate_sql, execute_sql, generate_answer

# Run with: python -m streamlit run app.py
st.set_page_config(page_title="Local Text-to-SQL", page_icon="🤖", layout="wide")


@st.cache_resource
def initialize_database():
    create_database()


st.title("🤖 Local Text-to-SQL")
st.write("Ask questions about the e-commerce dataset and get answers in plain English.")

try:
    initialize_database()
except Exception as error:
    st.error(f"Could not initialize the database: {error}")
    st.stop()

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

if prompt := st.chat_input("Example: How many customers are older than 40?"):
    question = prompt.strip()
    if question:
        history = list(st.session_state.messages)
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        display_message(user_message)
        assistant_message = {"role": "assistant"}
        try:
            with st.spinner("Checking the data with Qwen..."):
                sql = generate_sql(question, history=history)
                assistant_message["sql"] = sql
                if not validate_sql(sql):
                    raise ValueError("The generated SQL was rejected by the safety validator.")
                result = execute_sql(sql)
                assistant_message["result"] = result
                assistant_message["content"] = generate_answer(question, sql, result)
        except Exception as error:
            assistant_message.update(
                content=f"I couldn't answer this question: {error}", error=True
            )
        st.session_state.messages.append(assistant_message)
        display_message(assistant_message)
