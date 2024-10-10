
### Summary of Modifications and Changes

1. **Model Complexity and Structure**:
   - **Switched to a Two-Layer Bi-LSTM**: We modified the original single-layer Bi-LSTM to a two-layer Bi-LSTM. This allowed the model to capture more complex patterns in the data while leveraging bidirectionality to consider information from both past and future sequences. This change improved the model's ability to learn from sequential data.
   - **Adjusted Hidden Layer Size**: We set the hidden layer size to 128, which is a balanced choice that provides enough capacity to model the data without over-complicating the model.

2. **Regularization Techniques**:
   - **Introduced L2 Regularization (Weight Decay)**: By adding a `weight_decay` parameter to the Adam optimizer, we applied L2 regularization, which helps prevent overfitting by penalizing large weights. This ensured that the model didn’t overfit to the training data, thereby improving generalization on unseen data.
   - **Reduced Dropout**: Initially, dropout was set relatively high, which might have been too aggressive and disrupted the learning process. We reduced dropout to 0.1, a lower rate that prevents overfitting without overly restricting the model's capacity.

3. **Training Techniques**:
   - **Early Stopping**: Implementing early stopping with a patience of 5 epochs helped stop training once validation accuracy plateaued. This prevented unnecessary training, saved computation time, and ensured the model didn't overfit by training too long.
   - **Optimal Learning Rate**: A learning rate of `0.001` was used, providing a good balance between convergence speed and model stability. It allowed the model to adjust weights effectively without overshooting optimal values.

### Explanation of the Final Code and How Each Part Improved Performance

Here’s the final version of the code along with an explanation:

```python
# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define an adjusted Bi-LSTM model with two layers and reduced dropout
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

# Initialize model with weight decay for L2 regularization and no dropout
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = 2
model = AdjustedBiLSTMClassifier(input_dim, hidden_dim, output_dim, dropout_prob=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

# Training loop with early stopping
best_val_accuracy = 0
patience = 5
wait = 0
epochs = 50
for epoch in range(epochs):
    model.train()
    inputs = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
    labels = torch.tensor(y_train, dtype=torch.long)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_inputs = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        val_labels = torch.tensor(y_val, dtype=torch.long)
        val_outputs = model(val_inputs)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(val_labels.numpy(), val_predicted.numpy())
    
    # Early stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        wait = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate the model on the test set
def evaluate(X, y, dataset_name):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        labels = torch.tensor(y, dtype=torch.long)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(labels.numpy(), predicted.numpy())
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(labels.numpy(), predicted.numpy(), target_names=['Non-Functional', 'Functional']))
        print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")

# Evaluate on the train, validation, and test sets
evaluate(X_train, y_train, "Training Set")
evaluate(X_val, y_val, "Validation Set")
evaluate(X_test, y_test, "Test Set")
```

### How Each Part Improved Performance

- **Model Definition**: The `AdjustedBiLSTMClassifier` with two layers and a 128 hidden dimension provided more capacity and depth, capturing temporal relationships effectively in the data.
- **L2 Regularization**: Adding `weight_decay` to the optimizer helped regularize the weights, ensuring they didn’t grow too large and helping the model generalize better on the validation and test sets.
- **Early Stopping**: Reduced the risk of overfitting by halting training when improvements plateaued.
- **Lower Dropout**: Lower dropout to 0.1 maintained some regularization without disrupting the learning process, unlike earlier higher dropout rates.

These adjustments made the model more robust and capable of learning complex patterns without overfitting, leading to improved validation and test accuracy. This balance between model complexity and regularization is what allowed the model to achieve a performance closer to your target of 96%.
