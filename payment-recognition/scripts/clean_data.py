import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the preprocessed data
data_path = '../data/preprocessed/preprocessed_data.csv'
df = pd.read_csv(data_path)

# Step 2: Drop the 'description' column
df = df.drop(columns=['description'])

# Step 3: Preprocess other columns
# Completely remove spaces from the 'name' column
df['name'] = df['name'].apply(lambda x: x.replace(' ', ''))

# Standardize 'IBAN' format
df['IBAN'] = df['IBAN'].str.upper().replace(' ', '')

# Additional preprocessing steps for other columns can be added here

# Step 4: Label encode the 'name' and 'IBAN' columns
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])
df['IBAN'] = le.fit_transform(df['IBAN'])

# Save the LabelEncoder parameters
encoder_filename = '../data/scaler_params/label_encoders.joblib'
joblib.dump(le, encoder_filename)

# Step 5: Save the preprocessed data to a new CSV file
output_path = '../data/clean/cleaned_data.csv'
df.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
