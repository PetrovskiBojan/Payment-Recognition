import pandas as pd
from transformers import pipeline
import os

# Step 1: Load the last 1000 rows from the data
data_path = '../data/raw/generated_data.csv'
df = pd.read_csv(data_path)

# Select the last 1000 rows
last_1000_rows = df.tail(1000)

# Initialize the QA model with the new model name
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2", tokenizer="deepset/minilm-uncased-squad2")

# Step 2: Define a function for model prediction
def predict_reference(description):
    # Define the question and context for the model
    question = "What is the reference number?"
    context = description
    
    # Perform the question answering task
    result = qa_model(question=question, context=context)
    
    # Return the predicted reference and its confidence score
    return result['answer'], result['score']

# Step 3: Apply the function to each row and update the DataFrame
for index, row in last_1000_rows.iterrows():
    description = row['description']
    reference, score = predict_reference(description)
    print("Processing row:", index + 1, "Reference:", reference, "Score:", str(score))
    
    # Update the reference field if the model's confidence is above 90%
    if score > 0.2:
        last_1000_rows.at[index, 'reference'] = str(reference)  # Explicitly cast to string if necessary
        last_1000_rows.at[index, 'description'] = ""  # Clear the description

# Step 4: Append the preprocessed data to the existing clean_data.csv
output_path = '../data/preprocessed/preprocessed_data.csv'
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    final_df = pd.concat([existing_df, last_1000_rows], ignore_index=True)
else:
    final_df = last_1000_rows.copy()

final_df.to_csv(output_path, index=False)
