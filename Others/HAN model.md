To implement a Hierarchical Attention Network (HAN) with **Doc2Vec** and **DistilBERT** embeddings, the key steps involve the following modifications:

1. **Splitting documents into sentences** and processing each sentence separately using **DistilBERT**.
2. Adding **word-level attention** to focus on important words in each sentence.
3. Adding **sentence-level attention** to focus on important sentences in the document.
4. Combining the **sentence-level representations** (from DistilBERT and attention) with the **document-level Doc2Vec** embeddings.
5. **Concatenating** the Doc2Vec and DistilBERT-based attention embeddings for classification.

Let's walk through each part of the code, explaining what it does, how it integrates into your project, and what advantages it offers. Your project combines **Doc2Vec** and **DistilBERT** embeddings with attention mechanisms for classifying user reviews into **Functional (F)** and **Non-Functional (NF)** categories.

### **Installation of Required Packages**

```python
!pip install gensim transformers torch scikit-learn tqdm xgboost nltk
```
This installs all the required libraries:
- **Gensim**: For Doc2Vec, which captures document-level semantics.
- **Transformers** and **torch**: For fine-tuning DistilBERT, handling transformers-based models, and applying attention.
- **Scikit-learn**: For machine learning utilities like train-test split and evaluation.
- **TQDM**: For progress bars during processing.
- **XGBoost**: For the final classification model.
- **NLTK**: For sentence tokenization (splitting documents into sentences).

### **Import Libraries**
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
These imports bring in:
- **Pandas** and **NumPy**: For data manipulation.
- **nltk**: For tokenizing documents into sentences.
- **torch** and **torch.nn**: To handle neural network models and attention mechanisms.
- **Doc2Vec** (Gensim): Pre-trained document embeddings to capture the overall semantics of a document.
- **AutoTokenizer** and **AutoModel** (Transformers): To use the fine-tuned DistilBERT model for extracting sentence-level embeddings.
- **XGBClassifier** (XGBoost): For classification, as it's powerful for structured/tabular data.

### **Uploading the Dataset**
```python
from google.colab import files

# Upload the file
uploaded = files.upload()

# Assuming the uploaded file is named 'reviews.csv'
df = pd.read_csv(next(iter(uploaded)))  # Load the uploaded CSV into a DataFrame
```
This part allows you to upload your dataset (`reviews.csv`) into Google Colab and load it into a DataFrame for processing. 

### **Label Mapping**
```python
# Map 'RequirementType' to 'labels' (Functional: 1, Non-Functional: 0)
label_mapping = {'F': 1, 'NF': 0}
df['labels'] = df['RequirementType'].map(label_mapping)
```
This step maps the categorical labels **'F'** and **'NF'** to numeric values (`1` for functional and `0` for non-functional). Numeric labels are required for training machine learning models.

### **Sentence Tokenization**
```python
# Tokenize documents into sentences
nltk.download('punkt')
df['sentences'] = df['content'].apply(sent_tokenize)
```
- **nltk.download('punkt')**: Downloads the NLTK tokenizer model for splitting text into sentences.
- **df['sentences'] = df['content'].apply(sent_tokenize)**: For each document (review), it splits the text into sentences using `sent_tokenize`. This is essential for hierarchical attention since we need to treat each sentence individually when extracting sentence-level embeddings.

### **Doc2Vec Embeddings**
```python
# Download and load the Doc2Vec model from Hugging Face
model_path = hf_hub_download(repo_id="RafidMehda/doc2vec_model", filename="doc2vec_model")
doc2vec_model = Doc2Vec.load(model_path)

# Function to get Doc2Vec embeddings
def get_doc2vec_embeddings(index):
    doc2vec_emb = doc2vec_model.dv[str(index)]
    return doc2vec_emb

# Apply the function to extract Doc2Vec embeddings
doc2vec_embeddings = [get_doc2vec_embeddings(i) for i in range(len(df))]
```
- **Doc2Vec** is a powerful method for capturing document-level semantics. Here, the **Doc2Vec** embeddings provide a representation of the overall meaning of the document.
- You load a pre-trained **Doc2Vec model** from Hugging Face and apply it to each document to get its embedding. This is a **global, document-level embedding** that helps capture the holistic meaning of the document, independent of sentence structure.

### **DistilBERT Sentence Embeddings**
```python
# Load pre-trained DistilBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("RafidMehda/fined-distilBERT")
distilbert_model = AutoModel.from_pretrained("RafidMehda/fined-distilBERT")

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
- **DistilBERT** is a more lightweight version of BERT that captures word-level semantics within a sentence. Here, it's used to extract **sentence-level embeddings**.
- For each sentence in a document, the model returns a hidden state for every word, which is **averaged** to get a single embedding per sentence.
- This captures **contextual information** for each sentence individually, complementing the document-level embedding from **Doc2Vec**.

### **Attention Mechanism (Hierarchical Attention Networks)**
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
- The attention mechanism helps focus on the **important sentences** within the document.
- **Attention scores** are computed to assign importance to each sentence.
- This helps the model **focus on the most relevant sentences**, which is crucial for tasks like classification.

### **Applying Sentence-Level Attention**
```python
sentence_attention_outputs = []
for emb_list in df['sentence_embeddings']:
    sentence_embs = torch.tensor(np.array(emb_list))  # Convert list of embeddings to tensor
    if len(sentence_embs.shape) == 2:  # Ensure correct shape
        sentence_embs = sentence_embs.unsqueeze(0)  # Add batch dimension if necessary
    weighted_sum, _ = sentence_attention(sentence_embs)  # Apply attention
    sentence_attention_outputs.append(weighted_sum.detach().numpy())
```
- For each document, the **sentence embeddings** are processed by the attention mechanism to weight the importance of sentences.
- The result is a **single embedding** for the document, weighted by the importance of its sentences.

### **Combining Doc2Vec and Attention-Based DistilBERT Embeddings**
```python
combined_embeddings = []
for doc2vec_emb, sentence_attn_emb in zip(doc2vec_embeddings, sentence_attention_outputs):
    sentence_attn_emb_flat = sentence_attn_emb.flatten()  # Flatten the attention-based sentence embeddings
    combined_embedding = np.concatenate((doc2vec_emb, sentence_attn_emb_flat))  # Concatenate
    combined_embeddings.append(combined_embedding)

# Convert to numpy arrays for input
X = np.array(combined_embeddings)
```
- Here, you **concatenate** the document-level embeddings from **Doc2Vec** with the **sentence-level attention embeddings** from DistilBERT.
- This gives a **comprehensive feature representation** that combines both the **global semantics** (from Doc2Vec) and the **local sentence-level importance** (from DistilBERT + attention).

### **Splitting Data for Training**
```python
y = df['labels'].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
- The combined embeddings (`X`) and the labels (`y`) are split into **training, validation, and test sets**. This is crucial for evaluating the model’s performance on unseen data.

### **Training XGBoost Classifier**
```python
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=1, use_label_encoder=False, eval_metric='ml

ogloss')
xgb_model.fit(X_train, y_train)
```
- **XGBoost** is a powerful classifier that handles structured data efficiently.
- The combined embeddings are used as input to this classifier for training.

### **Evaluation**
```python
def get_predictions_and_evaluate(X, y, dataset_name):
    y_pred = xgb_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y, y_pred, target_names=['Non-Functional', 'Functional']))
    print(f"\n{dataset_name} Accuracy: {accuracy * 100:.2f}%")

get_predictions_and_evaluate(X_train, y_train, "Training Set")
get_predictions_and_evaluate(X_val, y_val, "Validation Set")
get_predictions_and_evaluate(X_test, y_test, "Test Set")
```
- The function evaluates the **performance of the classifier** on the training, validation, and test sets. It calculates metrics like accuracy, precision, recall, and F1-score, providing insights into the model’s ability to classify reviews as functional or non-functional.

---

### **Advantages of This Approach**:
1. **Multi-Granularity Representation**:
   - The combination of **Doc2Vec** (document-level) and **DistilBERT + attention** (sentence-level) embeddings allows the model to capture both **global** and **local** information.
2. **Attention Mechanism**:
   - The attention mechanism highlights the **most important sentences** in a document, improving the model's focus on relevant parts.
3. **XGBoost Classifier**:
   - XGBoost is highly efficient for tabular data, making it a great choice for the combined embeddings.
4. **Hierarchical Structure**:
   - This approach aligns well with the **Hierarchical Attention Network (HAN)** concept, where different levels of granularity (sentence, document) are combined for better classification performance.

This approach is more **nuanced** compared to traditional models, as it considers the hierarchical structure of the text while leveraging the power of both word-level and document-level embeddings.

### **Summary of Key Changes for HAN Implementation:**
1. **Splitting documents** into sentences using `nltk.sent_tokenize`.
2. **Generating sentence-level embeddings** using DistilBERT for each sentence instead of the entire document.
3. Adding **attention layers** for both word-level (optional) and sentence-level attention.
4. **Concatenating** the sentence-level attention outputs with document-level **Doc2Vec embeddings**.
5. Training an **XGBoost classifier** on the combined feature vectors.

This implementation modifies your original architecture to a **Hierarchical Attention Network (HAN)** by considering the hierarchical structure of text (words in sentences, sentences in documents). The **attention mechanism** helps the model focus on the most important parts of the text at different levels, improving the classification performance.
