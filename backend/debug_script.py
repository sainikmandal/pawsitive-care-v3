import pandas as pd
from pathlib import Path

# Make sure this path points to the root of your project
PROJECT_ROOT = Path(__file__).parent.parent # If this script is in backend/
# Or more directly if you know the full path:
# PROJECT_ROOT = Path("/Users/sainik/workspace/github.com/sainikmandal/pawsitive-care-v3/")

DATASET_PATH = PROJECT_ROOT / "animal_disease_dataset - animal_disease_dataset.csv.csv"

if not DATASET_PATH.exists():
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
else:
    print(f"Inspecting dataset: {DATASET_PATH}")
    df_train_actual = pd.read_csv(DATASET_PATH)
    print("\nUnique values in ACTUAL TRAINING DATA (after potential lowercase/strip):")

    symptom_cols = ['Symptom 1', 'Symptom 2', 'Symptom 3']
    for col in ['Animal'] + symptom_cols:
        if col in df_train_actual.columns:
            print(f"\n--- Unique values for column: {col} ---")
            # Apply similar cleaning as in your preprocess_data
            unique_vals = df_train_actual[col].astype(str).str.lower().str.strip().unique()
            print(sorted(list(unique_vals))) # Sorted for easier comparison
            print(f"Number of unique values: {len(unique_vals)}")

    # Check temperature ranges and resulting categories
    if 'Temperature' in df_train_actual.columns:
        print("\n--- Temperature Analysis ---")
        df_train_actual['Temperature_Processed'] = pd.to_numeric(df_train_actual['Temperature'], errors='coerce').fillna(0)
        temp_cats = pd.cut(
            df_train_actual['Temperature_Processed'],
            bins=[0, 100, 101, 102, 103, 104, float('inf')],
            labels=['very_low', 'low', 'normal', 'high', 'very_high', 'extreme'],
            include_lowest=True, right=False
        ).astype(str)
        print("Unique Temperature Categories in training data:")
        print(sorted(list(temp_cats.unique())))