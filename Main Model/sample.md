To incorporate an attention mechanism into your hybrid model that combines Doc2Vec and DistilBERT embeddings, we'll use an attention mechanism to learn how to weigh the combined embeddings before passing them to the classifier.

Here’s a full implementation that includes:

1. **Fine-tuning DistilBERT** with attention scores used for weighted token embeddings.
2. **Doc2Vec embeddings** used as before.
3. **Hybrid model with attention mechanism** to weigh both embeddings (Doc2Vec + DistilBERT).
4. **Classifier** (XGBoost or a simple neural network) to classify the reviews as functional (F) or non-functional (NF).

### Full Code:

```python
# Install required libraries
!pip install gensim transformers torch scikit-learn tqdm xgboost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel
from gensim.models import Doc2Vec
from tqdm import tqdm
import torch
import torch.nn as nn
from xgboost import XGBClassifier

# Define the Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        # Linear layer to compute attention weights
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, combined_embeddings):
        # Compute raw attention scores
        attention_scores = self.attention_weights(combined_embeddings)
        attention_scores = torch.softmax(attention_scores, dim=1)  # Normalize scores
        
        # Compute weighted sum of embeddings
        weighted_output = torch.sum(combined_embeddings * attention_scores, dim=1)
        return weighted_output

# Load the dataset into a DataFrame
df = pd.read_csv('labeled_reviews.csv')

# Map 'RequirementType' to 'labels' (Functional: 1, Non-Functional: 0)
label_mapping = {'F': 1, 'NF': 0}
df['labels'] = df['RequirementType'].map(label_mapping)

# Check if the 'labels' column was created correctly
print(df[['RequirementType', 'labels']].head())

# Load the pre-trained Doc2Vec model from Hugging Face or local storage
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="RafidMehda/doc2vec_model", filename="doc2vec_model")
doc2vec_model = Doc2Vec.load(model_path)

# Function to extract Doc2Vec embeddings
def get_doc2vec_embeddings(index):
    doc2vec_emb = doc2vec_model.dv[str(index)]
    return doc2vec_emb

# Extract Doc2Vec embeddings for each document in the dataset
doc2vec_embeddings = [get_doc2vec_embeddings(i) for i in range(len(df))]

# Load the fine-tuned DistilBERT model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("RafidMehda/fined-distilBERT")
hf_model = AutoModel.from_pretrained("RafidMehda/fined-distilBERT")

# Function to extract fine-tuned DistilBERT embeddings with attention
def get_finetuned_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = hf_model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # Token embeddings
        attention_weights = outputs.attentions  # Get attention weights (if available)
        # Compute weighted embedding using attention scores
        avg_attention = torch.mean(attention_weights[-1], dim=1)  # Use attention weights from the last layer
        weighted_embedding = torch.sum(last_hidden_state * avg_attention.unsqueeze(-1), dim=1)
        return weighted_embedding.squeeze().numpy()

# Generate embeddings using the fine-tuned DistilBERT model for the dataset
finetuned_embeddings = [get_finetuned_embeddings(doc) for doc in df['content']]

# Combine Doc2Vec and DistilBERT embeddings
combined_embeddings = [np.concatenate((doc2vec_emb, finetuned_emb)) for doc2vec_emb, finetuned_emb in zip(doc2vec_embeddings, finetuned_embeddings)]

# Convert the combined embeddings into a PyTorch tensor
combined_embeddings = torch.tensor(combined_embeddings)

# Initialize attention layer with the input size being the combined embeddings' dimension
attention_layer = AttentionLayer(input_dim=combined_embeddings.size(1))

# Apply the attention mechanism to the combined embeddings
weighted_embeddings = attention_layer(combined_embeddings)

# Convert weighted embeddings to numpy for compatibility with scikit-learn classifiers
X = weighted_embeddings.detach().numpy()
y = df['labels'].values

# Split data into train, validation, and test sets (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train XGBoost classifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Function to compute predictions and evaluate performance
def get_predictions_and_evaluate(X, y, dataset_name):
    y_pred = xgb_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y, y_pred, target_names=['Non-Functional', 'Functional']))
    print(f"\n{dataset_name} Accuracy: {accuracy * 100:.2f}%")

# Evaluate on the train, validation, and test sets
get_predictions_and_evaluate(X_train, y_train, "Training Set")
get_predictions_and_evaluate(X_val, y_val, "Validation Set")
get_predictions_and_evaluate(X_test, y_test, "Test Set")
```

### Key Components of the Code:

1. **Attention Layer:**
   - The `AttentionLayer` class takes the combined Doc2Vec and DistilBERT embeddings and computes attention scores. These scores are then used to weight the combined embeddings.

2. **Doc2Vec Embeddings:**
   - The Doc2Vec embeddings are extracted as in your original pipeline using the pre-trained model.

3. **DistilBERT Embeddings with Attention:**
   - The `get_finetuned_embeddings()` function retrieves the embeddings from the fine-tuned DistilBERT model and applies attention weights to emphasize important tokens.

4. **Combining Embeddings:**
   - The Doc2Vec and DistilBERT embeddings are concatenated, forming a hybrid representation of each review.
   - The `AttentionLayer` is applied to these combined embeddings to learn which dimensions of the embedding should be emphasized for the classification task.

5. **XGBoost Classifier:**
   - After applying the attention mechanism, the weighted embeddings are passed to an XGBoost classifier to predict whether a review is functional or non-functional.

### **Advantages of Using Attention in This Hybrid Model:**

- **Token-Level Focus:** The attention mechanism on DistilBERT embeddings allows the model to give more weight to important tokens, learning which words are more indicative of functional or non-functional requirements.
- **Combination of Contextual and Document-Level Representations:** By using attention on the combined Doc2Vec and BERT embeddings, the model learns to balance document-level context (from Doc2Vec) and fine-grained word-level context (from BERT) to make more accurate predictions.

### Next Steps:
- **Tune Hyperparameters:** You can adjust the parameters of the XGBoost classifier, DistilBERT fine-tuning, or Doc2Vec embeddings to further improve model performance.
- **Model Interpretation:** You can analyze the attention scores to understand which tokens and embeddings the model focuses on when making decisions, providing insight into the words that signify functional vs. non-functional requirements.

This approach leverages attention mechanisms effectively, providing more focused, contextually relevant embeddings to the classifier for better performance on your classification task.


Certainly! Let's break down the code **line by line**, focusing on what each part does in the context of your **hybrid NLP model** project, where you're combining **Doc2Vec** and **DistilBERT** embeddings and introducing an **attention mechanism** for improved classification.

---

### 1. **Installing Required Libraries**

```python
!pip install gensim transformers torch scikit-learn tqdm xgboost
```
- This line installs the required Python packages:
  - **gensim:** For working with the **Doc2Vec** model.
  - **transformers:** To work with the **DistilBERT** model from Hugging Face.
  - **torch:** The backend used for PyTorch, needed for deep learning and attention mechanisms.
  - **scikit-learn:** For preprocessing, splitting datasets, and evaluating models.
  - **tqdm:** A progress bar library that helps visualize long-running operations (e.g., tokenizing data).
  - **xgboost:** The XGBoost classifier for the final classification step.

---

### 2. **Defining the Attention Layer**

```python
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        # Linear layer to compute attention weights
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, combined_embeddings):
        # Compute raw attention scores
        attention_scores = self.attention_weights(combined_embeddings)
        attention_scores = torch.softmax(attention_scores, dim=1)  # Normalize scores
        
        # Compute weighted sum of embeddings
        weighted_output = torch.sum(combined_embeddings * attention_scores, dim=1)
        return weighted_output
```
- **AttentionLayer Class:** Defines a custom neural network layer using PyTorch’s `nn.Module`. The layer computes attention weights for the combined embeddings.
  - `__init__(self, input_dim)`: This is the constructor, initializing a linear layer that will calculate attention scores. The `input_dim` is the size of the combined Doc2Vec and DistilBERT embeddings.
  - `self.attention_weights = nn.Linear(input_dim, 1)`: This creates a linear layer that transforms the input (combined embeddings) into a single attention score per embedding dimension.
  - `forward(self, combined_embeddings)`: This method computes the attention scores for the embeddings and applies the attention mechanism.
    - **attention_scores = self.attention_weights(combined_embeddings):** Calculates raw attention scores by passing the combined embeddings through the linear layer.
    - **attention_scores = torch.softmax(attention_scores, dim=1):** Normalizes the raw attention scores using the **softmax** function so that they sum to 1. This step ensures the attention scores can be interpreted as probabilities, weighting the embeddings proportionally to their importance.
    - **weighted_output = torch.sum(combined_embeddings * attention_scores, dim=1):** Multiplies the combined embeddings by their corresponding attention scores and sums them up. This gives a weighted sum of embeddings, emphasizing important parts of the input.

---

### 3. **Loading the Dataset**

```python
df = pd.read_csv('labeled_reviews.csv')
```
- This loads your labeled dataset (`labeled_reviews.csv`) using **pandas**, which contains user reviews that have already been labeled as either **Functional (F)** or **Non-Functional (NF)**.

---

### 4. **Mapping Requirement Type to Labels**

```python
label_mapping = {'F': 1, 'NF': 0}
df['labels'] = df['RequirementType'].map(label_mapping)
```
- **label_mapping:** This dictionary maps **Functional (F)** reviews to the integer 1, and **Non-Functional (NF)** reviews to 0.
- **df['labels'] = df['RequirementType'].map(label_mapping):** This applies the mapping to the 'RequirementType' column of the DataFrame, creating a new column `labels` with the corresponding numerical labels. This is necessary because most machine learning algorithms, including XGBoost, require numeric input.

---

### 5. **Loading Pre-Trained Doc2Vec Model**

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="RafidMehda/doc2vec_model", filename="doc2vec_model")
doc2vec_model = Doc2Vec.load(model_path)
```
- **hf_hub_download:** This function downloads the pre-trained **Doc2Vec** model from your Hugging Face repository.
- **Doc2Vec.load(model_path):** This loads the pre-trained **Doc2Vec** model from the downloaded path. Doc2Vec will provide document-level embeddings for each review, capturing the global context.

---

### 6. **Extracting Doc2Vec Embeddings**

```python
def get_doc2vec_embeddings(index):
    doc2vec_emb = doc2vec_model.dv[str(index)]
    return doc2vec_emb

doc2vec_embeddings = [get_doc2vec_embeddings(i) for i in range(len(df))]
```
- **get_doc2vec_embeddings(index):** This function retrieves the Doc2Vec embedding for a specific document by its index in the DataFrame.
  - `doc2vec_model.dv[str(index)]`: This gets the document vector (embedding) for the document at position `index`.
- **doc2vec_embeddings = [get_doc2vec_embeddings(i) for i in range(len(df))]:** This line iterates through all reviews in the dataset and extracts the corresponding Doc2Vec embeddings, storing them in the list `doc2vec_embeddings`.

---

### 7. **Loading DistilBERT Model and Tokenizer**

```python
tokenizer = AutoTokenizer.from_pretrained("RafidMehda/fined-distilBERT")
hf_model = AutoModel.from_pretrained("RafidMehda/fined-distilBERT")
```
- **AutoTokenizer.from_pretrained:** This loads the tokenizer associated with the fine-tuned **DistilBERT** model from your Hugging Face repository.
  - The tokenizer is responsible for converting text (user reviews) into tokenized inputs that can be fed into the DistilBERT model.
- **AutoModel.from_pretrained:** This loads the fine-tuned **DistilBERT** model itself, which will be used to extract contextualized word embeddings from the input text.

---

### 8. **Extracting DistilBERT Embeddings with Attention**

```python
def get_finetuned_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = hf_model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # Token embeddings
        attention_weights = outputs.attentions  # Get attention weights (if available)
        avg_attention = torch.mean(attention_weights[-1], dim=1)  # Last layer attention
        weighted_embedding = torch.sum(last_hidden_state * avg_attention.unsqueeze(-1), dim=1)
        return weighted_embedding.squeeze().numpy()
```
- **tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128):** This tokenizes the input text, converting it into token IDs and adding padding/truncating to a maximum length of 128 tokens. The output is a PyTorch tensor.
- **with torch.no_grad():** This disables gradient calculation since we’re only extracting embeddings (no backpropagation needed).
- **outputs = hf_model(**inputs):** Passes the tokenized input into the fine-tuned DistilBERT model. The model returns:
  - **last_hidden_state:** A matrix containing token embeddings for each word in the review.
  - **attention_weights:** The attention scores for each token (if available).
- **avg_attention = torch.mean(attention_weights[-1], dim=1):** Averages the attention weights from the last attention layer. This step collapses the multiple heads in multi-head attention into a single vector.
- **weighted_embedding = torch.sum(last_hidden_state * avg_attention.unsqueeze(-1), dim=1):** Applies the attention scores to weight the token embeddings, then sums them to produce a single embedding that represents the entire review.
- **return weighted_embedding.squeeze().numpy():** Converts the weighted embedding from PyTorch to a NumPy array.

---

### 9. **Generating Fine-tuned Embeddings for All Reviews**

```python
finetuned_embeddings = [get_finetuned_embeddings(doc) for doc in df['content']]
```
- This line iterates over all the review content in the dataset and extracts fine-tuned embeddings for each review using DistilBERT, storing the results in the list `finetuned_embeddings`.

---

### 10. **Combining Doc2Vec and DistilBERT Embeddings**

```python
combined_embeddings = [np.concatenate((doc2vec_emb, finetuned_emb)) for doc2vec_emb, finetuned_emb in zip(doc2vec_embeddings, finetuned_embeddings)]
combined_embeddings = torch.tensor(combined_embeddings)
```
- **np.concatenate((doc2vec_emb, finetuned_emb)):** For each review, this concatenates the **Doc2Vec** embedding with the **DistilBERT** embedding to create a single feature vector that includes both document-level and contextualized word-level information.
- **torch.tensor(combined_embeddings):** Converts the combined embeddings into a PyTorch tensor, which is necessary to apply the attention mechanism.

---

### 11. **Applying the Attention Layer**

```python
attention_layer = AttentionLayer(input_dim=combined_embeddings.size(1))
weighted_embeddings = attention_layer(combined_embeddings)
```
- **AttentionLayer(input_dim=combined_embeddings.size(1)):** Initializes the attention layer with the size

 of the combined embeddings (the number of features).
- **weighted_embeddings = attention_layer(combined_embeddings):** Passes the combined embeddings through the attention layer to compute the weighted sum of the embeddings, emphasizing important features.

---

### 12. **Preparing Data for XGBoost Classifier**

```python
X = weighted_embeddings.detach().numpy()
y = df['labels'].values
```
- **weighted_embeddings.detach().numpy():** Converts the PyTorch tensor with the weighted embeddings into a NumPy array, which is required for compatibility with the XGBoost classifier.
- **y = df['labels'].values:** Extracts the labels (target variable) from the DataFrame.

---

### 13. **Splitting Data into Train, Validation, and Test Sets**

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
- **train_test_split(X, y, test_size=0.3):** Splits the data into a training set (70%) and a temporary set (30%).
- **train_test_split(X_temp, y_temp, test_size=0.5):** Further splits the temporary set into validation (15%) and test sets (15%).

---

### 14. **Training the XGBoost Classifier**

```python
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
```
- **XGBClassifier():** Initializes the XGBoost classifier with specific hyperparameters:
  - **n_estimators=100:** The number of trees in the ensemble (100 boosting rounds).
  - **learning_rate=0.1:** Controls the contribution of each tree to the final prediction.
  - **max_depth=6:** Limits the depth of the trees to control overfitting.
  - **use_label_encoder=False:** Disables label encoding since we’re directly using integer labels.
  - **eval_metric='mlogloss':** The evaluation metric is **multi-class log loss**.
- **xgb_model.fit(X_train, y_train):** Trains the XGBoost model on the training data.

---

### 15. **Evaluating the Model**

```python
def get_predictions_and_evaluate(X, y, dataset_name):
    y_pred = xgb_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y, y_pred, target_names=['Non-Functional', 'Functional']))
    print(f"\n{dataset_name} Accuracy: {accuracy * 100:.2f}%")
```
- **get_predictions_and_evaluate():** This function evaluates the model on a given dataset (train, validation, or test):
  - **xgb_model.predict(X):** Predicts labels for the input data.
  - **accuracy_score(y, y_pred):** Computes the accuracy of the predictions.
  - **classification_report(y, y_pred):** Provides precision, recall, and F1-scores for each class.

---

### 16. **Evaluating on Train, Validation, and Test Sets**

```python
get_predictions_and_evaluate(X_train, y_train, "Training Set")
get_predictions_and_evaluate(X_val, y_val, "Validation Set")
get_predictions_and_evaluate(X_test, y_test, "Test Set")
```
- These lines call the evaluation function for the training set, validation set, and test set, printing the classification report and accuracy for each.

---

### Summary

- The code combines **Doc2Vec** and **DistilBERT** embeddings, applying an **attention mechanism** to emphasize important features before classification.
- The **XGBoost classifier** uses the weighted embeddings to predict whether a review is **functional (F)** or **non-functional (NF)**.

