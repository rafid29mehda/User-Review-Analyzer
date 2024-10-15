![Uploading Bi-lstm.jpg…]()



Let me provide a detailed breakdown of the Bi-LSTM classifier architecture based on the image.

### Layers of the Bi-LSTM Architecture

#### 1. **Input Layer**
- The **input layer** consists of a sequence of input vectors (`X(t-1)`, `X(t)`, `X(t+1)`).
- Each input represents the hybrid embeddings that combine **Doc2Vec** and **DistilBERT** features, capturing both document-level and contextualized word-level information.
- The **sequence** allows the Bi-LSTM to understand temporal dependencies between the features across time steps.

#### 2. **BiLSTM1 Hidden Layer**
- **First Bi-LSTM Layer (BiLSTM1)**:
  - This layer consists of two **LSTM cells**, one for the **forward direction** (left to right) and one for the **backward direction** (right to left).
  - The forward LSTM processes the input from `X(t-1)` to `X(t+1)`, and the backward LSTM processes the input in the reverse order.
  - The outputs from both LSTMs at each time step are concatenated to form the hidden representation (`h1(t)` and `ĥ1(t)`).
  - **Bi-LSTM** captures context from both past and future directions, which is especially useful in understanding the semantics of reviews in this context.
  
#### 3. **Dropout Layer**
- **Dropout Layer** is introduced after the first Bi-LSTM hidden layer:
  - The purpose of dropout is to **prevent overfitting** by randomly deactivating some neurons during training.
  - This leads to a more generalized model, as the network doesn't rely too heavily on any particular neurons.

#### 4. **BiLSTM2 Hidden Layer**
- **Second Bi-LSTM Layer (BiLSTM2)**:
  - Similar to the first Bi-LSTM layer, this layer has **forward and backward LSTMs** that take the output from the previous layer (after dropout) as input.
  - By using a second Bi-LSTM layer, the model can better capture more complex temporal dynamics in the sequence data.
  - The outputs at each time step are denoted as (`h2(t)` and `ĥ2(t)`).
  - This stacked Bi-LSTM architecture provides the model with a greater capability to model long-term dependencies between the features.

#### 5. **Classification Layer**
- **Final Classification Layer**:
  - After the output from the second Bi-LSTM layer, we apply an additional operation (`σ`) which seems to represent a non-linear activation function (like ReLU or Tanh).
  - This representation is then passed to the **classification layer**, which consists of a **fully connected (dense) layer**.
  - The dense layer outputs the logits for the two possible classes, **Functional (F)** or **Non-Functional (NF)**.

#### 6. **SoftMax Layer**
- **Softmax Layer**:
  - The final output is passed through a **softmax layer**.
  - The softmax function produces a probability distribution over the two classes (F and NF), allowing the network to decide the category for each input.
  - It provides confidence scores indicating the likelihood of the input belonging to each of the classes.

### Summary
- The architecture consists of **two stacked Bi-LSTM layers**. Each Bi-LSTM is made of **forward and backward LSTMs**, allowing the model to capture both past and future dependencies in the data.
- The **dropout layer** between the two Bi-LSTM layers helps to generalize the model by preventing overfitting.
- The **final dense classification layer**, followed by the **softmax function**, is used for categorizing the reviews into functional or non-functional categories.

The design of two Bi-LSTM layers aims to learn complex relationships in the input data by combining both **contextual sequence** information and **global document features**. The **hierarchical nature** of the Bi-LSTM layers is crucial for your model to capture fine-grained details in a hierarchical structure, further reinforced by the combination of embeddings from **Doc2Vec** and **DistilBERT**.

This architecture diagram illustrates how temporal data (embeddings) is sequentially processed in both directions and how important mechanisms, like dropout and stacking, ensure robust and generalizable feature extraction before classification. It helps build a rich representation of the document, effectively leveraging both word-level and document-level information for classification tasks. 

Let me know if you need more specific details or explanations of any part of this architecture!
