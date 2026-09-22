import pandas as pd

df = pd.read_csv("data/ecommerce_cleaned.csv")
print(df.shape)
for col in df.columns:
    if df[col].nunique() <= 20:
        print(col, df[col].value_counts().to_dict())