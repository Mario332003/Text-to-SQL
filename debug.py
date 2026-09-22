import json
from main import (CSV_PATH, create_database, get_database_schema,
                  generate_analysis_queries, execute_sql)

print("CSV in use:", CSV_PATH)
ds = create_database()
print("DB in use:", ds["db_path"])

schema_json = get_database_schema(ds["schema_path"])
schema = json.loads(schema_json)
print("\nCOLUMNS:", [c["name"] for c in schema["columns"]])

total = execute_sql('SELECT COUNT(*) AS n FROM dataset', ds["db_path"])
print("\nTOTAL ROWS:", int(total["n"][0]))

print("\nCOLUMNS WITH FEW DISTINCT VALUES:")
for c in schema["columns"]:
    q = f'SELECT "{c["name"]}" AS v, COUNT(*) AS n FROM dataset GROUP BY 1 ORDER BY n DESC LIMIT 8'
    df = execute_sql(q, ds["db_path"])
    if len(df) < 8:
        print(" ", c["name"], dict(zip(df["v"], df["n"])))

print("\nPLANNED SQL for 'Analyze orders from 2025':")
for item in generate_analysis_queries("Analyze orders from 2025", schema_json,
                                      db_path=ds["db_path"]):
    print("-", item["purpose"])
    print("  ", item["sql"])