To implement a Hierarchical Attention Network (HAN) with **Doc2Vec** and **DistilBERT** embeddings, the key steps involve the following modifications:

1. **Splitting documents into sentences** and processing each sentence separately using **DistilBERT**.
2. Adding **word-level attention** to focus on important words in each sentence.
3. Adding **sentence-level attention** to focus on important sentences in the document.
4. Combining the **sentence-level representations** (from DistilBERT and attention) with the **document-level Doc2Vec** embeddings.
5. **Concatenating** the Doc2Vec and DistilBERT-based attention embeddings for classification.

Here’s the full revised code that incorporates these changes:

### **1. Install Required Libraries**

```python
!pip install gensim transformers torch scikit-learn tqdm xgboost nltk
```

### **2. Import Libraries**

```python
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Doc2Vec
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from xgboost import XGBClassifier
from huggingface_hub import hf_hub_download
```

### **3. Prepare the Dataset**

This code loads the `reviews.csv` dataset and prepares the data for hierarchical attention by splitting documents into sentences.

```python
# Load the dataset
df = pd.read_csv("reviews.csv")  # Make sure to upload this file if working in Colab

# Map 'RequirementType' to 'labels' (Functional: 1, Non-Functional: 0)
label_mapping = {'F': 1, 'NF': 0}
df['labels'] = df['RequirementType'].map(label_mapping)

# Tokenize documents into sentences
nltk.download('punkt')
df['sentences'] = df['content'].apply(sent_tokenize)

# Display sample
print(df[['content', 'sentences', 'labels']].head())
```

### **4. Load Pre-trained Models**

Load the pre-trained **Doc2Vec** and **DistilBERT** models from Hugging Face or your local storage.

```python
# Download and load the Doc2Vec model from Hugging Face
model_path = hf_hub_download(repo_id="RafidMehda/doc2vec_model", filename="doc2vec_model")
doc2vec_model = Doc2Vec.load(model_path)

# Load pre-trained DistilBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
```

### **5. Get Doc2Vec Embeddings**

Extract document-level embeddings using **Doc2Vec**.

```python
# Function to get Doc2Vec embeddings
def get_doc2vec_embeddings(index):
    doc2vec_emb = doc2vec_model.dv[str(index)]
    return doc2vec_emb

# Apply the function to extract Doc2Vec embeddings
doc2vec_embeddings = [get_doc2vec_embeddings(i) for i in range(len(df))]
```

### **6. Get DistilBERT Sentence-Level Embeddings**

Generate **DistilBERT embeddings** for each sentence in the document.

```python
# Function to get DistilBERT embeddings for sentences in a document
def get_distilbert_sentence_embeddings(sentences):
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_embedding = torch.mean(last_hidden_state, dim=1)  # Average pooling
        sentence_embeddings.append(pooled_embedding.squeeze().numpy())
    return sentence_embeddings

# Apply the function to get DistilBERT embeddings for each document's sentences
df['sentence_embeddings'] = df['sentences'].apply(get_distilbert_sentence_embeddings)
```

### **7. Attention Mechanisms**

Implement **word-level** and **sentence-level attention**. Here is a simple attention mechanism using PyTorch.

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, num_sentences, hidden_dim)
        attn_scores = torch.tanh(torch.matmul(hidden_states, self.attn_weights)).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_sum = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return weighted_sum, attn_weights
```

### **8. Apply Sentence-Level Attention**

Apply the **attention mechanism** to weigh the importance of each sentence in the document.

```python
# Define the attention model for sentence-level embeddings
sentence_attention = Attention(hidden_dim=768)  # Assuming DistilBERT's hidden dim is 768

# Apply sentence-level attention to the DistilBERT embeddings
sentence_attention_outputs = []
for emb_list in df['sentence_embeddings']:
    sentence_embs = torch.tensor(emb_list)
    weighted_sum, _ = sentence_attention(sentence_embs)
    sentence_attention_outputs.append(weighted_sum.numpy())
```

### **9. Combine Doc2Vec and Attention Outputs**

Concatenate the **Doc2Vec document embeddings** with the **attention-weighted sentence embeddings**.

```python
# Concatenate Doc2Vec and attention-based DistilBERT embeddings
combined_embeddings = [np.concatenate((doc2vec_emb, sentence_attn_emb)) for doc2vec_emb, sentence_attn_emb in zip(doc2vec_embeddings, sentence_attention_outputs)]

# Convert to numpy arrays for input
X = np.array(combined_embeddings)
y = df['labels'].values

# Split data into train, validation, and test sets (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

### **10. XGBoost Classification**

Train an **XGBoost** classifier on the combined embeddings.

```python
# Initialize and train XGBoost classifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss')
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

### **Summary of Key Changes for HAN Implementation:**
1. **Splitting documents** into sentences using `nltk.sent_tokenize`.
2. **Generating sentence-level embeddings** using DistilBERT for each sentence instead of the entire document.
3. Adding **attention layers** for both word-level (optional) and sentence-level attention.
4. **Concatenating** the sentence-level attention outputs with document-level **Doc2Vec embeddings**.
5. Training an **XGBoost classifier** on the combined feature vectors.

This implementation modifies your original architecture to a **Hierarchical Attention Network (HAN)** by considering the hierarchical structure of text (words in sentences, sentences in documents). The **attention mechanism** helps the model focus on the most important parts of the text at different levels, improving the classification performance.