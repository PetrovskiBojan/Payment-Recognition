import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Step 1: Load the data
data_path = '../data/raw/initial_data.csv'
df = pd.read_csv(data_path)

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
for index, row in df.iterrows():
    description = row['description']
    reference, score = predict_reference(description)
    print("reference:", reference, "score:", str(score))
    
    # Update the reference field if the model's confidence is above 90%
    if score > 0.2:
        df.at[index, 'reference'] = str(reference)  # Explicitly cast to string if necessary
        df.at[index, 'description'] = ""  # Clear the description

# Step 4: Save the preprocessed data to a new CSV file
output_path = '../data/preprocessed/initial_preprocessed_data.csv'
df.to_csv(output_path, index=False)
