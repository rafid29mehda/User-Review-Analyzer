The error indicates a shape mismatch in the attention mechanism. Specifically, it occurs because the combined embedding size (868 in this case) does not match the expected hidden size (768 for DistilBERT). This mismatch arises because we concatenate the DistilBERT and Doc2Vec embeddings, leading to an increased dimension.

To resolve this, adjust the attention mechanism to accommodate the concatenated embedding size. Here's how:

### Solution
1. **Update the Attention Layer Initialization** to match the combined embedding size.
2. **Verify the Combined Embedding Size** to ensure it matches the concatenated vector.

### Steps to Fix
Adjust the attention mechanism to take the combined embedding size as an input parameter. Replace the hidden size with the combined size when initializing the `Attention` layer.

Here’s the corrected code:

```python
# Adjusted Attention mechanism to take combined embedding size
class Attention(nn.Module):
    def __init__(self, combined_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Parameter(torch.Tensor(combined_dim, 1))
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, hidden_states):
        attn_scores = torch.tanh(torch.matmul(hidden_states, self.attn_weights)).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_sum = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return weighted_sum, attn_weights

# Use the combined dimension for initialization
combined_dim = 768 + doc2vec_model.vector_size  # Adjust based on DistilBERT + Doc2Vec size
sentence_attention = Attention(combined_dim)

# Rest of the code remains the same...

# Example: Test the attention weights visualization
sample_sentence = "The application should allow users to reset their password easily."
tokens, attention_weights = combined_embeddings_with_attention(sample_sentence)
plot_word_attention(tokens, attention_weights)

# Redefine Bi-LSTM with the combined embedding size
class AdjustedBiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super(AdjustedBiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, 0, :]  # Only use the first time step
        logits = self.fc(lstm_out)
        return self.softmax(logits)

# Initialize and load Bi-LSTM classifier model
input_dim = combined_dim  # Use combined dimension here
hidden_dim = 128
output_dim = 2
model = AdjustedBiLSTMClassifier(input_dim, hidden_dim, output_dim, dropout_prob=0.1)
model.load_state_dict(torch.load('best_model.pt'))
```

### Explanation of Changes
- **Attention Initialization**: The `Attention` class now takes the combined embedding size (`combined_dim`) when defining the attention weights.
- **Bi-LSTM Input Dimension**: Updated to use the concatenated size of Doc2Vec and DistilBERT embeddings.

This should resolve the matrix multiplication error, allowing the attention mechanism to process the combined embeddings correctly. Run the updated code, and it should work for visualizing attention weights and classifying sentences without errors.
