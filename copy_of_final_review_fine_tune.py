# -*- coding: utf-8 -*-
"""Copy of Final_Review_Fine_tune.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JQE1JtUYQz3-qwTtrM6Wdc7OIMIAG7jl
"""

!pip install transformers datasets torch

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from google.colab import files

# Upload the CSV file
uploaded = files.upload()

# Load the uploaded file into a DataFrame
df = pd.read_csv(next(iter(uploaded)))

# Map labels to integers (Functional: 1, Non-Functional: 0)
label_mapping = {'F': 1, 'NF': 0}
df['labels'] = df['RequirementType'].map(label_mapping)

# Split the dataset into train and test
train_df, test_df = train_test_split(df[['content', 'labels']], test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

from transformers import DistilBertTokenizer

# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['content'], padding="max_length", truncation=True)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 1: Get predictions from the model on the test set
predictions = trainer.predict(test_dataset)

# Step 2: Convert logits to predicted class
preds = np.argmax(predictions.predictions, axis=-1)

# Step 3: Calculate accuracy
accuracy = accuracy_score(test_df['labels'], preds)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 4: Calculate precision, recall, and F1-score
precision = precision_score(test_df['labels'], preds, average='weighted')
recall = recall_score(test_df['labels'], preds, average='weighted')
f1 = f1_score(test_df['labels'], preds, average='weighted')

# Print Precision, Recall, F1-score
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Step 5: Optional - Print full classification report for more detailed metrics
print("Classification Report:\n")
print(classification_report(test_df['labels'], preds, target_names=['Non-Functional', 'Functional'], zero_division=1))

# Save the fine-tuned model
trainer.save_model('./app_review_model')

# Save the tokenizer files
tokenizer.save_pretrained('./app_review_model')

!pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

from huggingface_hub import HfApi

# Upload the entire directory to Hugging Face
api = HfApi()
api.upload_folder(
    folder_path='./app_review_model',  # Path to the folder with the model and tokenizer files
    repo_id='RafidMehda/app_review_model',  # Your model repository on Hugging Face
    repo_type='model'
)

