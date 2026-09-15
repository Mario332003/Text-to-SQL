import streamlit as st

from main import (
    create_database,
    generate_sql,
    validate_sql,
    execute_sql
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Local Text-to-SQL",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# INITIALIZE DATABASE
# ---------------------------------------------------------

@st.cache_resource
def initialize_database():
    create_database()


initialize_database()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🤖 Local Text-to-SQL")

st.write(
    "Ask a question about the e-commerce dataset "
    "and the AI will convert it into SQL."
)

st.divider()


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

question = st.text_area(
    "Ask a question",
    placeholder="Example: How many customers are older than 40?",
    height=100
)


# ---------------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------------

if st.button(
    "Generate SQL",
    type="primary",
    use_container_width=True
):

    # Check that the user entered something
    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            # ---------------------------------------------
            # STEP 1: GENERATE SQL
            # ---------------------------------------------

            with st.spinner(
                "Generating SQL using Qwen..."
            ):

                sql = generate_sql(
                    question
                )


            # ---------------------------------------------
            # STEP 2: DISPLAY GENERATED SQL
            # ---------------------------------------------

            st.subheader(
                "Generated SQL"
            )

            st.code(
                sql,
                language="sql"
            )


            # ---------------------------------------------
            # STEP 3: VALIDATE SQL
            # ---------------------------------------------

            if not validate_sql(sql):

                st.error(
                    "The generated SQL was rejected "
                    "by the safety validator."
                )

                st.stop()


            # ---------------------------------------------
            # STEP 4: EXECUTE SQL
            # ---------------------------------------------

            with st.spinner(
                "Executing query..."
            ):

                result = execute_sql(
                    sql
                )


            # ---------------------------------------------
            # STEP 5: DISPLAY RESULT
            # ---------------------------------------------

            st.subheader(
                "Result"
            )

            if result.empty:

                st.info(
                    "No results found."
                )

            else:

                st.dataframe(
                    result,
                    use_container_width=True
                )


        except Exception as error:

            st.error(
                f"Error: {error}"
            )