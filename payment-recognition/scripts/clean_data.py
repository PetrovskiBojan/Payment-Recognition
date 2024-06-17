import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths to the files
preprocessed_data_path = 'data/preprocessed/preprocessed_data.csv'
initial_preprocessed_data_path = 'data/preprocessed/initial_preprocessed_data.csv'
generated_data_path = 'data/raw/generated_data.csv'
initial_raw_data_path = 'data/raw/initial_data.csv'
reference_data_path = 'data/validate/reference_data.csv'

# Function to read and append last 100 rows from one CSV to another
def append_last_100_rows(source_path, target_path):
    # Read the source file and take the last 100 rows
    source_df = pd.read_csv(source_path)
    last_100_rows = source_df.tail(100)
    
    # Read the target file
    target_df = pd.read_csv(target_path)
    
    # Append the last 100 rows to the target dataframe
    updated_df = pd.concat([target_df, last_100_rows], ignore_index=True)
    
    # Save the updated dataframe back to the target file
    updated_df.to_csv(target_path, index=False)

# Save the initial preprocessed data to the reference path before appending
initial_preprocessed_data_df = pd.read_csv(initial_preprocessed_data_path)
initial_preprocessed_data_df.to_csv(reference_data_path, index=False)

# Append last 100 rows from preprocessed_data.csv to initial_preprocessed_data.csv
append_last_100_rows(preprocessed_data_path, initial_preprocessed_data_path)

# Append last 100 rows from generated_data.csv to initial_raw_data.csv
append_last_100_rows(generated_data_path, initial_raw_data_path)

# Load the preprocessed data again after appending the last 100 rows
df = pd.read_csv(preprocessed_data_path)

# Preprocess other columns
# Completely remove spaces from the 'name' column
df['name'] = df['name'].apply(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

# Standardize 'IBAN' format
df['IBAN'] = df['IBAN'].str.upper().replace(' ', '')

# Process 'description' column
df['description'] = df['description'].astype(str).fillna('unknown_description')

# Mark descriptions as -1 if they do not match the reference or if they are not purely numeric
def process_description(row):
    try:
        # If description matches reference or is purely numeric, return the integer value of description
        if row['description'] == str(row['reference']) or row['description'].isdigit():
            return int(row['description'])
        else:
            # If description is neither matching reference nor numeric, return -1
            return int(row['reference'])
    except ValueError:
        # If conversion to int fails, return -1
        return -1

df['description'] = df.apply(process_description, axis=1)

# Convert non-numeric references to -1
def process_reference(row):
    try:
        # Attempt to convert reference to integer; if fails, return -1
        return int(row['reference'])
    except ValueError:
        # If conversion to int fails, return -1
        return -1

df['reference'] = df.apply(process_reference, axis=1)

# Label encode the 'name', 'IBAN', and 'description' columns
le_name = LabelEncoder()
df['name'] = le_name.fit_transform(df['name'])

le_iban = LabelEncoder()
df['IBAN'] = le_iban.fit_transform(df['IBAN'])

le_description = LabelEncoder()
df['description'] = le_description.fit_transform(df['description'])

# Save the LabelEncoder parameters
joblib.dump(le_name, 'data/scaler_params/label_encoder_name.joblib')
joblib.dump(le_iban, 'data/scaler_params/label_encoder_iban.joblib')
joblib.dump(le_description, 'data/scaler_params/label_encoder_description.joblib')

# Separate unique references and duplicates
duplicates = df[df.duplicated(subset='reference', keep=False)]
unique = df.drop_duplicates(subset='reference', keep=False)

# Ensure at least one record of each duplicate is in the train set
train_data = unique.copy()
test_data = pd.DataFrame()

for ref in duplicates['reference'].unique():
    ref_duplicates = duplicates[duplicates['reference'] == ref]
    if not ref_duplicates.empty:
        # Add the first occurrence to the train set
        train_data = pd.concat([train_data, ref_duplicates.iloc[[0]]])
        # Add the remaining occurrences to the test set
        test_data = pd.concat([test_data, ref_duplicates.iloc[1:]])

# Filter out rows with reference -1 from test data
test_data = test_data[test_data['reference'] != -1]

# Save the separated data to new CSV files
train_data_path = 'data/clean/train_data.csv'
test_data_path = 'data/clean/test_data.csv'
train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)

print(f"Unique references and one of each duplicate saved to {train_data_path}")
print(f"Remaining duplicates saved to {test_data_path} (excluding rows with reference -1)")

print("Data processing complete.")
