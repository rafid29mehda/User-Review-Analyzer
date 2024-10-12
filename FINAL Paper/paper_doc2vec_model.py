# -*- coding: utf-8 -*-
"""paper_doc2vec_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q2KoeFdc_EVzGSpkQXo_-Lk7Ecku19D4
"""

# Import necessary libraries
!pip install gensim

import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from google.colab import files
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Upload the CSV file in Google Colab
uploaded = files.upload()  # Opens a file dialog for file upload

# Step 2: Load the dataset into a DataFrame
df = pd.read_csv(next(iter(uploaded)))  # Load the uploaded file into a DataFrame

# Step 3: Download NLTK resources
nltk.download('punkt')

# Step 4: Map labels to integers (Functional: 1, Non-Functional: 0)
label_mapping = {'F': 1, 'NF': 0}
df['labels'] = df['RequirementType'].map(label_mapping)

# Step 5: Prepare tagged documents for Doc2Vec
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(df['content'])]

# Initialize the Doc2Vec model
model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)

# Build the vocabulary from the tagged documents
model.build_vocab(tagged_data)

# Train the Doc2Vec model
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Extract document vectors
doc_vectors = [model.dv[str(i)] for i in range(len(tagged_data))]

# Split data into training, validation, and testing sets (60-20-20 split)
X_train, X_temp, y_train, y_temp = train_test_split(doc_vectors, df['labels'], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train a logistic regression classifier on the training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate on the training set
y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Set Classification Report:\n")
print(classification_report(y_train, y_train_pred, target_names=['Non-Functional', 'Functional']))
print(f"Training Set Accuracy: {train_accuracy * 100:.2f}%\n")

# Evaluate on the validation set
y_val_pred = classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred, target_names=['Non-Functional', 'Functional']))
print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%\n")

# Evaluate on the test set
y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred, target_names=['Non-Functional', 'Functional']))
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%\n")

# Save the Doc2Vec model
# model.save('doc2vec_model')

# Step 1: Train and Save the Doc2Vec Model
model.save('doc2vec_model')

# Step 2: Install huggingface_hub
!pip install huggingface_hub

# Step 3: Login to Hugging Face
from huggingface_hub import notebook_login
notebook_login()

# Step 4: Upload the model to Hugging Face
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi()

# Upload the model to your Hugging Face repository
api.upload_folder(
    folder_path='./',  # Folder path where 'doc2vec_model' is located
    repo_id='RafidMehda/paper_doc2vec_model',  # Your Hugging Face repository name
    repo_type='model'  # Specify that it's a model repository
)