import json
import agent2.main2 as m
print("LOADED FROM:", m.__file__)

q = "Analyze how wage and value differ across positions and nationalities."
ds = m.create_database()
schema = m.get_database_schema(ds["schema_path"])
plan = m.generate_analysis_queries(q, schema, db_path=ds["db_path"])
executed = m.run_analysis_queries(plan, ds["db_path"])

for e in executed:
    print("\nQUERY:", e["sql"])
    print("KEYS:", list(e.keys()))          # expect 'extremes' for the big nationality query
    if "extremes" in e:
        print(json.dumps(e["extremes"], indent=1)[:800])

print("\nANSWER:\n", m.narrate_multi_query_analysis(q, executed))