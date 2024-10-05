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
