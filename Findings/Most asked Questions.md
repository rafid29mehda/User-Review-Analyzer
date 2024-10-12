
---

### **1. Why did we use sentence-level embeddings to create the Hierarchical Attention Network (HAN)?**

**Understanding the Nature of Text Data:**

- **Hierarchical Structure of Documents**: Natural language documents inherently have a hierarchical structure. Words form sentences, and sentences form documents. Capturing this hierarchy can lead to better representation and understanding of the text.

**Benefits of Sentence-Level Embeddings in HAN:**

- **Capturing Context at Multiple Levels**: By using sentence-level embeddings, HAN captures context at both the word level (within sentences) and the sentence level (within documents). This dual-level attention allows the model to focus on the most informative words in a sentence and the most important sentences in a document.
  
- **Improved Representation**: Sentence-level embeddings help in summarizing the essential information in each sentence, which is especially useful when dealing with long documents or reviews where not all sentences contribute equally to the overall meaning.
  
- **Attention Mechanism Efficiency**: Applying attention at the sentence level reduces computational complexity compared to applying attention across all words in the document. It allows the model to weigh sentences based on their contribution to the overall classification task.

**Why Sentence-Level Embeddings Matter in Your Model:**

- **Handling Long Reviews**: User reviews can vary in length and often contain multiple sentences that contribute differently to whether the review is functional (F) or non-functional (NF). Sentence-level embeddings help in isolating these contributions.

- **Facilitating Hierarchical Attention**: The HAN architecture is designed to work with hierarchical inputs. Sentence-level embeddings are essential to apply attention mechanisms at both word and sentence levels, aligning with the hierarchical nature of language.

---

### **2. Why can't we combine fine-tuned Doc2Vec and fine-tuned DistilBERT directly?**

**Differences Between Doc2Vec and DistilBERT:**

- **Doc2Vec**:
  - Provides **document-level embeddings**.
  - Generates a fixed-size vector representing the entire document.
  - Captures the overall semantic meaning but may miss finer-grained contextual details.

- **DistilBERT**:
  - Provides **contextualized word-level embeddings**.
  - Generates embeddings for each word, taking into account its context within the sentence.
  - Excels at capturing local contextual information but doesn't inherently produce a single embedding for the entire document.

**Challenges in Direct Combination:**

- **Mismatch in Granularity**:
  - Doc2Vec embeddings represent entire documents, while DistilBERT embeddings represent words or tokens.
  - Directly combining them without considering the hierarchical structure can lead to misalignment in the information they capture.

- **Dimensionality Differences**:
  - The embeddings from Doc2Vec and DistilBERT may have different dimensions, making direct concatenation non-trivial without proper alignment.

- **Loss of Contextual Hierarchy**:
  - Simply combining the two embeddings at the document level might overlook the valuable hierarchical relationships within the text (i.e., how sentences and words contribute differently).

**Advantages of Using HAN Instead:**

- **Hierarchical Processing**:
  - HAN processes the text in a way that respects its hierarchical structure, first encoding words into sentence embeddings and then sentences into document embeddings.
  
- **Attention Mechanisms at Multiple Levels**:
  - Allows the model to focus on the most relevant words within sentences and the most important sentences within documents.

- **Effective Integration of Embeddings**:
  - By applying attention mechanisms, HAN can effectively integrate the strengths of both Doc2Vec and DistilBERT embeddings, leading to better performance.

---

### **3. Given that DistilBERT can perform word-level and sentence-level multi-head attention, why did we combine the models in this way?**

**Understanding DistilBERT's Capabilities:**

- **DistilBERT's Attention Mechanism**:
  - Uses self-attention to create contextualized embeddings for words within sentences.
  - Primarily focuses on capturing dependencies between tokens in the input sequence.

- **Limitations in Hierarchical Context**:
  - While DistilBERT captures word-level context effectively, it doesn't inherently model the hierarchical structure of documents (i.e., relationships between sentences).

**Reasons for Combining Models in This Way:**

- **Leveraging Complementary Strengths**:
  - **Doc2Vec** captures global semantic information at the document level.
  - **DistilBERT** provides rich contextual embeddings at the word level.
  - Combining both allows the model to utilize both global and local contextual information.

- **Enhancing Hierarchical Modeling**:
  - By using DistilBERT to obtain sentence embeddings (through averaging or pooling word embeddings), and then applying attention at the sentence level, we can model inter-sentence relationships more effectively.

- **Custom Attention Mechanisms**:
  - The HAN architecture introduces custom attention layers that can be fine-tuned specifically for the task, potentially outperforming the pre-trained attention in DistilBERT when applied at higher hierarchical levels.

**Why Not Rely Solely on DistilBERT's Attention:**

- **Pre-Trained Attention May Not Suffice**:
  - The attention mechanisms in DistilBERT are trained on generic language modeling tasks and may not capture the specific hierarchical relationships important for classifying reviews into F and NF categories.

- **Task-Specific Adaptation**:
  - By designing a custom HAN, we can tailor the attention mechanisms to focus on the most relevant parts of the text for our specific classification task.

---

### **4. How did we create the Hierarchical Attention Network (HAN), and how is it helping?**

**Creation of the HAN:**

- **Sentence Embedding Generation**:
  - **DistilBERT** is used to obtain embeddings for each sentence:
    - Each sentence is tokenized and passed through DistilBERT.
    - Word embeddings within a sentence are aggregated (e.g., via mean pooling) to create a sentence embedding.

- **Sentence-Level Attention Mechanism**:
  - An attention layer is applied to the sentence embeddings:
    - Assigns weights to sentences based on their importance to the overall meaning of the document.
    - Outputs a weighted sum of sentence embeddings, producing a document representation that emphasizes important sentences.

- **Combining with Doc2Vec Embeddings**:
  - The attention-weighted sentence embeddings are concatenated with the **Doc2Vec** embeddings:
    - Merges the global document-level semantic information from Doc2Vec with the attention-enhanced local context from DistilBERT.

- **Classification with Bi-LSTM**:
  - The combined embeddings are fed into a Bi-directional Long Short-Term Memory (Bi-LSTM) network:
    - Captures sequential patterns and dependencies in the data.
    - Outputs are passed through a fully connected layer and softmax activation for classification.

**How HAN is Helping:**

- **Capturing Hierarchical Structure**:
  - By processing text at both the word and sentence levels, HAN models the hierarchical nature of language more effectively than flat models.

- **Enhanced Representation**:
  - The attention mechanisms allow the model to focus on the most informative words and sentences, improving the quality of the representations.

- **Improved Classification Performance**:
  - Combining the strengths of both Doc2Vec and DistilBERT, along with hierarchical attention and Bi-LSTM, leads to better performance in classifying reviews into F and NF categories.

- **Handling Long and Complex Texts**:
  - HAN is particularly beneficial for long reviews where important information may be scattered across sentences.

**Visual Representation of the HAN Architecture:**

1. **Word-Level Encoding (within DistilBERT)**:
   - Input sentences are tokenized, and word embeddings are generated using DistilBERT.
   - Word embeddings are aggregated to form sentence embeddings.

2. **Sentence-Level Attention**:
   - An attention mechanism assigns weights to sentence embeddings.
   - Produces a context vector that emphasizes important sentences.

3. **Combining with Doc2Vec Embeddings**:
   - The context vector from the attention mechanism is concatenated with the Doc2Vec embedding of the document.
   - Results in a comprehensive representation that includes both local (sentence-level) and global (document-level) information.

4. **Bi-LSTM Classifier**:
   - The combined embeddings are fed into a Bi-LSTM network.
   - Captures temporal dependencies and patterns.
   - Outputs class probabilities after passing through a fully connected layer and softmax.

---

### **Additional Insights:**

- **Why Not Use DistilBERT Alone?**
  - While DistilBERT is powerful, using it alone might not capture the global semantic nuances that Doc2Vec provides.
  - The combination ensures that both the macro (document-level) and micro (word and sentence-level) contexts are considered.

- **Benefits of Hierarchical Attention:**
  - **Interpretability**: Attention weights can be inspected to understand which sentences or words contributed most to the classification, providing insights into model decisions.
  - **Flexibility**: HAN can be adapted to different tasks by modifying the attention mechanisms or incorporating other embeddings.

- **Model Complexity vs. Performance:**
  - The architecture balances complexity and performance:
    - **Complexity**: Introducing attention mechanisms and combining multiple embeddings increases computational requirements.
    - **Performance**: The potential gain in classification accuracy and robustness often justifies the added complexity.

---

### **Conclusion:**

By using sentence-level embeddings and creating a Hierarchical Attention Network, you effectively capture both local and global contextual information in your model. This approach leverages the strengths of fine-tuned Doc2Vec and DistilBERT models while addressing their individual limitations when used in isolation. The HAN architecture, combined with a Bi-LSTM classifier, enhances the model's ability to classify user reviews accurately by focusing on the most relevant parts of the text at multiple hierarchical levels.

---
