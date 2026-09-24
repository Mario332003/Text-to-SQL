import pandas as pd
import numpy as np

# --- update these two paths for your machine ---
INPUT_PATH = r"C:\Users\farah\New folder (2)\Text-to-SQL\fifa_eda_stats.csv"
OUTPUT_PATH = r"C:\Users\farah\New folder (2)\Text-to-SQL\fifa_eda_stats_cleaned.csv"
# -------------------------------------------------

df = pd.read_csv(INPUT_PATH)
print("Original shape:", df.shape)

# 1. Drop exact duplicate rows
before = df.shape[0]
df = df.drop_duplicates()
print(f"Duplicates removed: {before - df.shape[0]}")

# 2. Drop records missing Position + every skill rating together
#    (these players have almost no usable data — not worth imputing)
skill_cols = ['Crossing', 'Finishing', 'Dribbling', 'BallControl', 'GKDiving']
incomplete_mask = df[skill_cols].isnull().all(axis=1)
print(f"Dropping {incomplete_mask.sum()} incomplete player records")
df = df[~incomplete_mask].copy()

# 3. Convert money columns (€110.5M / €565K / €0) to numeric euros
def money_to_number(val):
    if pd.isnull(val):
        return np.nan
    val = str(val).replace('€', '').strip()
    if val.endswith('M'):
        return float(val[:-1]) * 1_000_000
    if val.endswith('K'):
        return float(val[:-1]) * 1_000
    try:
        return float(val)
    except ValueError:
        return np.nan

for col in ['Value', 'Wage', 'Release Clause']:
    df[col] = df[col].apply(money_to_number)

# Missing release clause -> no clause -> 0 (not median)
df['Release Clause'] = df['Release Clause'].fillna(0)

# 4. Convert Height (5'7) to cm, Weight (159lbs) to kg
def height_to_cm(val):
    if pd.isnull(val):
        return np.nan
    try:
        feet, inches = str(val).split("'")
        return round(int(feet) * 30.48 + int(inches) * 2.54, 1)
    except Exception:
        return np.nan

def weight_to_kg(val):
    if pd.isnull(val):
        return np.nan
    try:
        lbs = float(str(val).replace('lbs', '').strip())
        return round(lbs * 0.453592, 1)
    except Exception:
        return np.nan

df['Height_cm'] = df['Height'].apply(height_to_cm)
df['Weight_kg'] = df['Weight'].apply(weight_to_kg)
df = df.drop(columns=['Height', 'Weight'])

# 5. Loaned From: blank means "not on loan"
df['Loaned From'] = df['Loaned From'].fillna('Not Loaned')

# 6. Small pockets of missing categorical data -> explicit "Unknown"
for col in ['Club', 'Contract Valid Until', 'Position', 'Work Rate', 'Body Type', 'Preferred Foot']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# 7. Remaining numeric gaps -> median
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# 8. Joined date: missing -> "Unknown"
df['Joined'] = df['Joined'].fillna('Unknown')

print("\nFinal shape:", df.shape)
print("Remaining nulls:\n", df.isnull().sum()[df.isnull().sum() > 0])

df.to_csv(OUTPUT_PATH, index=False)
print("\nSaved cleaned CSV to:", OUTPUT_PATH)