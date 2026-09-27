import agent2.main2 as m
print("LOADED FROM:", m.__file__)

ds = m.create_database()
schema = m.get_database_schema(ds["schema_path"])
plan = m.generate_analysis_queries(
    "Analyze how wage and value differ across positions and nationalities.",
    schema, db_path=ds["db_path"],
)
print("\nFINAL PLAN:")
for q in plan:
    print("-", q["purpose"], "\n ", q["sql"])
print("EMPTY PLAN = fallback" if not plan else "")