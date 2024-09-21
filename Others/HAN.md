Final merged doc2vec and distilBERT code can serve as a strong starting point for implementing **Hierarchical Attention Networks (HAN)** with **DistilBERT** and **Doc2Vec** embeddings. However, some modifications are needed to achieve the hierarchical architecture and the attention mechanisms required by HAN.

Here’s a high-level outline of how you can modify the current code to implement HAN:

### Steps for HAN with Doc2Vec and DistilBERT:

1. **Document-Level Semantics with Doc2Vec:**
   - You can retain the **Doc2Vec** embeddings for capturing document-level representations, which will help the model understand the overall structure and semantic meaning of the text.

2. **Sentence-Level Features with DistilBERT:**
   - Instead of extracting DistilBERT embeddings for entire documents, you will need to tokenize and split each document into sentences and generate **DistilBERT embeddings** for each sentence. This will capture the finer-grained, sentence-level features that the HAN will weigh based on their importance.

3. **Attention Mechanism at Word Level:**
   - In a typical HAN, the attention mechanism is applied to focus on important words in a sentence. You can extract **DistilBERT embeddings** for individual tokens (words), and apply an attention mechanism to assign weights to each token to highlight important words.

4. **Sentence-Level Attention:**
   - After applying attention at the word level, you'll aggregate sentence embeddings (via attention) to compute the overall sentence importance. The HAN architecture uses another attention layer to weigh the importance of each sentence within the document.

5. **Concatenation and Classification:**
   - You will concatenate the sentence-level attention outputs with the document-level **Doc2Vec** embeddings.
   - The combined representation can then be passed through a classification layer (i.e., a dense layer with a softmax function) to classify the document as **Functional** or **Non-Functional**.

### Key Changes Required:

#### 1. **Splitting Documents into Sentences:**
You will need to split the text in the `content` column into individual sentences before generating DistilBERT embeddings.

You can use a sentence tokenizer, like `nltk.sent_tokenize`, to split each document into sentences.

```python
import nltk
nltk.download('punkt')

# Example of splitting text into sentences
sentences = nltk.sent_tokenize(df['content'][0])  # Split the first document into sentences
```

#### 2. **Generate DistilBERT Embeddings at the Sentence Level:**
Modify your `get_distilbert_embeddings` function to work on individual sentences rather than the entire document. You will create embeddings for each sentence.

```python
def get_distilbert_sentence_embeddings(sentences):
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_embedding = torch.mean(last_hidden_state, dim=1)  # Average pooling of token embeddings
        sentence_embeddings.append(pooled_embedding.squeeze().numpy())
    return sentence_embeddings
```

#### 3. **Implement the Attention Mechanism:**
You’ll need to implement two levels of attention:

- **Word-Level Attention**: Focuses on important words within a sentence.
- **Sentence-Level Attention**: Focuses on important sentences within a document.

For the attention mechanism, you can define custom PyTorch layers. A typical attention mechanism calculates a weighted sum of input representations using learnable attention weights.

Example of a simple attention mechanism in PyTorch:

```python
import torch.nn as nn
import torch.nn.functional as F

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

You can apply this attention mechanism first at the **word level** and then at the **sentence level**.

#### 4. **Combine Sentence and Document-Level Representations:**
Once you have sentence-level embeddings after applying attention, combine them with the **Doc2Vec** document-level embeddings:

```python
# Example of combining Doc2Vec and sentence embeddings
final_representation = torch.cat((doc2vec_embedding, sentence_level_attention_output), dim=-1)
```

#### 5. **Classification Layer:**
After obtaining the final combined representation, pass it through a classification layer:

```python
classification_layer = nn.Linear(combined_embedding_size, num_classes)
output = classification_layer(final_representation)
```

### Full Pipeline Overview:

1. **Doc2Vec Embeddings**: Obtain document-level embeddings using the pre-trained Doc2Vec model (as in the original code).
2. **Sentence Splitting**: Split each document into sentences.
3. **DistilBERT Embeddings**: For each sentence, generate DistilBERT embeddings at the word level.
4. **Word-Level Attention**: Apply attention over the word embeddings within each sentence.
5. **Sentence-Level Attention**: Aggregate sentence embeddings and apply attention over sentences.
6. **Concatenation**: Concatenate the document-level Doc2Vec embeddings with the attention-weighted sentence embeddings.
7. **Classification**: Use a final classification layer to classify the document as **Functional** or **Non-Functional**.

### Benefits of HAN:
- **Hierarchical Attention** allows the model to focus on both important words and sentences, enhancing its ability to capture nuanced information.
- Combining **Doc2Vec** (document-level semantics) with **DistilBERT** (sentence-level context) improves the model's capability to understand the text at multiple granularities.

### Conclusion:
You can definitely modify your existing code to implement a **Hierarchical Attention Network (HAN)** combining **Doc2Vec** and **DistilBERT** embeddings. The modifications mainly involve splitting the document into sentences, implementing word- and sentence-level attention, and then concatenating the results for classification.
