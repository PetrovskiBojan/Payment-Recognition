import pandas as pd
from transformers import pipeline
import os
import subprocess

# Step 1: Pull the latest data from DVC
def run_dvc_pull(data_directory):
    """Pull the latest data from DVC for the specified directory."""
    print("Pulling latest data from DVC...")
    try:
        subprocess.run(['dvc', 'pull', data_directory], check=True)
        print("Data pulled successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to pull data from DVC:", e)
        raise

# Specify the directory containing DVC-tracked data
data_directory = 'data'  

# Pull the latest data from DVC
run_dvc_pull(data_directory)

# Step 2: Load the last 1000 rows from the data
data_path = 'data/raw/generated_data.csv'
df = pd.read_csv(data_path)

# Select the last 1000 rows
last_100_rows = df.tail(100)

# Initialize the QA model with the new model name
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2", tokenizer="deepset/minilm-uncased-squad2")

# Step 3: Define a function for model prediction
def predict_reference(description):
    # Define the question and context for the model
    question = "What is the reference number?"
    context = description
    
    # Perform the question answering task
    result = qa_model(question=question, context=context)
    
    # Return the predicted reference and its confidence score
    return result['answer'], result['score']

# Step 4: Apply the function to each row and update the DataFrame
for index, row in last_100_rows.iterrows():
    description = row['description']
    reference, score = predict_reference(description)
    print("Processing row:", index + 1, "Reference:", reference, "Score:", str(score))
    
    # Update the reference field if the model's confidence is above 90%
    if score > 0.2:
        last_100_rows.at[index, 'reference'] = str(reference)  # Explicitly cast to string if necessary
        last_100_rows.at[index, 'description'] = ""  # Clear the description

# Step 5: Append the preprocessed data to the existing clean_data.csv
output_path = 'data/preprocessed/preprocessed_data.csv'
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    final_df = pd.concat([existing_df, last_100_rows], ignore_index=True)
else:
    final_df = last_100_rows.copy()

final_df.to_csv(output_path, index=False)
