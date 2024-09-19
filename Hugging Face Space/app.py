import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the fine-tuned model from Hugging Face
model_path = "RafidMehda/fin_review_model"  
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Mapping the labels to more meaningful names
label_map = {
    'LABEL_1': 'Functional',
    'LABEL_0': 'Non-Functional'
}

st.title("Review Classifier: Functional (F) vs Non-Functional (NF)")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload a CSV file with a 'Review' column", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Check if 'Review' column exists
    if 'Review' in df.columns:
        # Analyze each review with error handling
        def classify_review(review):
            try:
                # Get the label from the classifier and map it to a meaningful name
                label = classifier(review)[0]['label']
                return label_map.get(label, "Unknown")  # Use 'Unknown' if the label is not recognized
            except Exception as e:
                return f"Error: {str(e)}"

        # Apply the classifier to each review
        df['Classification'] = df['Review'].apply(classify_review)
        
        # Display the results
        st.write(df)

        # Provide a download button for the results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download classified results", data=csv, file_name='classified_reviews.csv', mime='text/csv')
    else:
        st.error("The CSV file must contain a 'Review' column.")
