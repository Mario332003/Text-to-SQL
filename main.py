import os
import re
import sqlite3

import pandas as pd
import ollama


# =========================================================
# CONFIGURATION
# =========================================================

CSV_PATH = "/Users/mario/Text-to-SQL/data/ecommerce_cleaned.csv"

DB_PATH = "/Users/mario/Text-to-SQL/data/ecommerce.db"

TABLE_NAME = "ecommerce"

MODEL_NAME = "qwen3:8b-q4_K_M"


# =========================================================
# DATABASE SCHEMA
# =========================================================

CREATE_TABLE_SQL = """
CREATE TABLE ecommerce (

    order_id TEXT,
    order_date TEXT,
    order_year INTEGER,
    order_month INTEGER,
    order_day INTEGER,
    order_hour INTEGER,
    order_minute INTEGER,
    order_second INTEGER,
    is_weekend INTEGER,

    order_status TEXT,
    return_reason TEXT,

    customer_id TEXT,
    customer_name TEXT,
    gender TEXT,
    age REAL,
    customer_segment TEXT,
    country TEXT,
    city TEXT,
    customer_loyalty_score REAL,
    total_orders_by_customer INTEGER,
    account_creation_date TEXT,

    product_id TEXT,
    product_name TEXT,
    category TEXT,
    sub_category TEXT,
    brand TEXT,
    product_rating_avg REAL,
    product_reviews_count INTEGER,
    stock_quantity INTEGER,

    unit_price_usd REAL,
    quantity REAL,
    discount_percent INTEGER,
    discount_amount_usd REAL,
    total_price_usd REAL,
    cost_usd REAL,
    profit_usd REAL,
    tax_usd REAL,
    currency TEXT,

    payment_method TEXT,
    payment_status TEXT,
    installment_plan INTEGER,

    shipping_method TEXT,
    shipping_cost_usd REAL,
    delivery_days INTEGER,
    shipping_country TEXT,
    warehouse_location TEXT,
    delivery_status TEXT,

    rating INTEGER,
    review_sentiment TEXT,
    customer_feedback TEXT,

    coupon_used INTEGER,
    coupon_code TEXT,
    campaign_source TEXT,

    device_type TEXT,
    traffic_source TEXT,
    session_duration_minutes REAL,
    pages_visited INTEGER,
    abandoned_cart_before INTEGER,

    fraud_risk_score REAL,
    profit_margin_percent REAL,
    order_priority TEXT,
    support_ticket_created INTEGER
);
"""


# =========================================================
# CREATE SQLITE DATABASE FROM CSV
# =========================================================

def create_database():

    # Reuse the database only when the expected table is present.
    if os.path.exists(DB_PATH):
        existing_connection = sqlite3.connect(DB_PATH)
        table_exists = existing_connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (TABLE_NAME,)
        ).fetchone()
        existing_connection.close()

        if table_exists:
            print("Database already exists.")
            return

    print("Creating SQLite database...")

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    # Create table
    cursor.execute(CREATE_TABLE_SQL)

    connection.commit()

    # Read large CSV in chunks
    chunk_size = 25000

    rows_loaded = 0

    for chunk in pd.read_csv(
        CSV_PATH,
        chunksize=chunk_size
    ):

        # Convert boolean columns to 0/1
        boolean_columns = [
            "is_weekend",
            "installment_plan",
            "coupon_used",
            "abandoned_cart_before",
            "support_ticket_created"
        ]

        for column in boolean_columns:
            if column in chunk.columns:
                chunk[column] = chunk[column].astype(
                    "Int64"
                )

        # Append to SQLite
        chunk.to_sql(
            TABLE_NAME,
            connection,
            if_exists="append",
            index=False
        )

        rows_loaded += len(chunk)

        print(
            f"Loaded {rows_loaded:,} rows..."
        )

    print("Creating indexes...")

    # Useful indexes for common analytical queries
    cursor.execute(
        """
        CREATE INDEX idx_order_date
        ON ecommerce(order_date);
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_customer_id
        ON ecommerce(customer_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_product_id
        ON ecommerce(product_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_category
        ON ecommerce(category);
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_brand
        ON ecommerce(brand);
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_country
        ON ecommerce(country);
        """
    )

    connection.commit()

    connection.close()

    print("Database created successfully.")


# =========================================================
# AUTOMATIC SCHEMA EXTRACTION
# =========================================================

def get_database_schema():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        f"PRAGMA table_info({TABLE_NAME})"
    )

    columns = cursor.fetchall()

    connection.close()

    schema = f"TABLE: {TABLE_NAME}\n\n"

    for column in columns:

        column_name = column[1]
        column_type = column[2]

        schema += (
            f"- {column_name} {column_type}\n"
        )

    return schema


# =========================================================
# GENERATE SQL USING QWEN
# =========================================================

def generate_sql(question):

    schema = get_database_schema()

    system_prompt = """
You are an expert Text-to-SQL system.

Your job is to convert natural-language questions
into valid SQLite SQL queries.

IMPORTANT RULES:

1. Use SQLite syntax only.

2. The database contains exactly one table:
   ecommerce

3. Use only columns that exist in the provided schema.

4. Never invent tables or columns.

5. Generate SELECT queries only.

6. Never generate:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   CREATE
   REPLACE
   TRUNCATE

7. Return ONLY the SQL query.

8. Do not use Markdown.

9. Do not wrap the SQL in ```sql.

10. Do not explain the query.

11. For Boolean columns:
    1 means TRUE
    0 means FALSE.

12. order_date is stored as:
    YYYY-MM-DD HH:MM:SS

13. account_creation_date is also stored as text.

14. Monetary values are primarily represented
    using USD columns.

15. Use appropriate aggregation functions
    such as:
    SUM()
    AVG()
    COUNT()
    MIN()
    MAX()

16. When calculating revenue or sales value,
    use total_price_usd unless the question
    clearly asks for another measure.

17. When calculating profit,
    use profit_usd.

18. When calculating number of orders,
    consider COUNT(*) or COUNT(DISTINCT order_id)
    according to the user's question.
"""

    user_prompt = f"""
DATABASE SCHEMA:

{schema}

USER QUESTION:

{question}

Generate the SQLite query.
"""

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        # Qwen3 normally supports thinking.
        # We don't need the reasoning trace here.
        think=False,

        stream=False,

        options={
            "temperature": 0
        }
    )

    sql = response.message.content

    return clean_sql(sql)


# =========================================================
# CLEAN MODEL OUTPUT
# =========================================================

def clean_sql(sql):

    sql = sql.strip()

    # Remove markdown fences if model ignores instruction
    sql = re.sub(
        r"^```sql",
        "",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"^```",
        "",
        sql
    )

    sql = re.sub(
        r"```$",
        "",
        sql
    )

    return sql.strip()


# =========================================================
# SQL SAFETY CHECK
# =========================================================

def validate_sql(sql):

    dangerous_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "REPLACE",
        "ATTACH",
        "DETACH",
        "PRAGMA"
    ]

    sql_upper = sql.upper()

    # Only SELECT / WITH queries allowed
    if not (
        sql_upper.startswith("SELECT")
        or sql_upper.startswith("WITH")
    ):
        return False

    for keyword in dangerous_keywords:

        pattern = rf"\b{keyword}\b"

        if re.search(
            pattern,
            sql_upper
        ):
            return False

    return True


# =========================================================
# EXECUTE GENERATED SQL
# =========================================================

def execute_sql(sql):

    # Open database read-only
    connection = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    try:

        dataframe = pd.read_sql_query(
            sql,
            connection
        )

        return dataframe

    finally:

        connection.close()


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    print("=" * 60)
    print("LOCAL TEXT-TO-SQL SYSTEM")
    print("=" * 60)

    # Step 1
    create_database()

    print()

    # Show schema
    schema = get_database_schema()

    print("DATABASE SCHEMA")
    print("-" * 60)

    print(schema)

    print("=" * 60)

    while True:

        print()

        question = input(
            "Ask a question about the dataset "
            "(or type 'exit'): "
        )

        if question.lower() in [
            "exit",
            "quit"
        ]:
            break

        print()

        print("Generating SQL...")

        try:

            sql = generate_sql(
                question
            )

            print()
            print("Generated SQL:")
            print("-" * 60)
            print(sql)
            print("-" * 60)

            # Safety validation
            if not validate_sql(sql):

                print(
                    "SQL rejected by safety validator."
                )

                continue

            print()

            print("Executing query...")

            result = execute_sql(
                sql
            )

            print()
            print("RESULT")
            print("-" * 60)

            if result.empty:

                print("No results found.")

            else:

                print(
                    result.to_string(
                        index=False
                    )
                )

            print("-" * 60)

        except Exception as error:

            print()
            print("ERROR:")
            print(error)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()