import pandas as pd
from transformers import pipeline

# Step 1: Load the data with 'reference' column as string to avoid .0 suffix
data_path = 'data/raw/initial_data.csv'
df = pd.read_csv(data_path, dtype={'reference': str})

# Initialize the QA model with the new model name
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2", tokenizer="deepset/minilm-uncased-squad2")

# Step 2: Define a function for model prediction
def predict_reference(description):
    if not description or pd.isna(description):  # Check if the description is empty or NaN
        return "", 0.0
    
    try:
        # Define the question and context for the model
        question = "What is the reference number?"
        context = description
        
        # Perform the question answering task
        result = qa_model(question=question, context=context)
        
        # Return the predicted reference and its confidence score
        return result['answer'], result['score']
    except Exception as e:
        print(f"Error processing description: {description}. Error: {e}")
        return "", 0.0

# Step 3: Apply the function to each row and update the DataFrame
for index, row in df.iterrows():
    description = row['description']
    reference, score = predict_reference(description)
    print("Reference:", reference, "Score:", str(score))
    
    # Update the reference field if the model's confidence is above 50%
    if score > 0.5:
        df.at[index, 'reference'] = str(reference)  # Store reference as string
        df.at[index, 'description'] = str(reference) 
# Step 4: Save the preprocessed data to a new CSV file
output_path = 'data/preprocessed/initial_preprocessed_data.csv'
df.to_csv(output_path, index=False)

print(f"Data has been processed and saved to {output_path}")
