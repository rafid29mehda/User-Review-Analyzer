### Presentation Script for Figure 2: Architecture of the Bi-LSTM Classification

---

**Slide Title:** Bi-LSTM Classification Architecture

---

**Introduction (10-15 seconds):**
"Now let’s dive into how the Bi-LSTM classification component processes the combined embeddings to classify software requirements into functional or non-functional categories. This figure illustrates the detailed workflow, from using combined embeddings as input to generating classification results."

---

**Step 1: Combined Embeddings as Input (10-15 seconds):**
"The process starts with the combined embeddings, which merge document-level embeddings from Doc2Vec and sentence-level embeddings from DistilBERT. These embeddings, rich in both global and local contextual information, are fed into this Bi-LSTM architecture for further processing."

---

**Step 2: Input Sequence and Sentence-Level Attention (15-20 seconds):**
"The first step involves breaking the input embeddings into sequences of words for each sentence, labeled as `S1, S2, ..., Sn`. Sentence-level attention is applied here to identify the most important sentences in the document. This ensures that the model prioritizes sentences that contribute the most to classification accuracy."

---

**Step 3: First Bi-LSTM Hidden Layer (20-25 seconds):**
"Next, these sequences are passed into the first Bidirectional Long Short-Term Memory, or Bi-LSTM, hidden layer. This layer processes the text in two directions: forward and backward. By doing so, it captures dependencies between words, both from past to future and future to past. This ensures that the model understands the context of each word within its surrounding sequence."

---

**Step 4: Word Attention Mechanism (15-20 seconds):**
"After the first Bi-LSTM layer, a word attention mechanism is applied. This mechanism assigns weights to individual words based on their importance. Words that are more relevant to the classification task are given higher attention scores, creating a weighted representation that highlights the key information."

---

**Step 5: Second Bi-LSTM Hidden Layer (20-25 seconds):**
"The output of the word attention mechanism is then passed into a second Bi-LSTM hidden layer. This layer further refines the representation by capturing higher-order dependencies and relationships in the sequence. Like the first layer, it processes data in both directions to ensure no contextual information is missed."

---

**Step 6: Dropout Layers (10-15 seconds):**
"Between these layers, dropout regularization is applied to prevent overfitting. By randomly removing some connections during training, dropout ensures that the model generalizes well to unseen data."

---

**Step 7: Attention Weight Calculation (15-20 seconds):**
"Following the second Bi-LSTM layer, attention weights are calculated. These weights represent the significance of each sequence and word, further refining the representation to focus only on the most important parts of the input."

---

**Step 8: SoftMax Classification Layer (10-15 seconds):**
"Finally, the refined representation is passed into a softmax classification layer. This layer generates probabilities over the two output classes: functional requirements, or FR, and non-functional requirements, or NFR. The highest probability determines the final classification."

---

**Conclusion (10-15 seconds):**
"To summarize, this Bi-LSTM architecture processes the combined embeddings using two layers of sequential modeling, enhanced by attention mechanisms at both the word and sentence levels. By combining contextual understanding with attention, it ensures accurate and interpretable classification of software requirements."

---

**Transition (Optional):**
"With this robust classification pipeline in mind, let’s now explore how the model performs and how it compares to other approaches."

---
