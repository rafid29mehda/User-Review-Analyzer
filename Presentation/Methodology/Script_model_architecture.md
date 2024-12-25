### Script for Explaining Figure 1 in a Presentation

---

**Slide Title:** Explaining the Hybrid Hierarchical Attention Network (H2AN) Architecture

---

**Introduction (10-15 seconds):**
"Let me introduce the architecture of the Hybrid Hierarchical Attention Network, or H2AN, which is the foundation of our proposed model for software requirements classification. This figure demonstrates how document-level and sentence-level embeddings are combined using advanced techniques to create a powerful classification model."

---

**Step 1: Input Data (10-15 seconds):**
"We start with the labeled dataset, which includes user reviews. These reviews are processed through two main pipelines: Doc2Vec for document-level embeddings and DistilBERT for sentence-level embeddings."

---

**Step 2: Fine-Tuning Doc2Vec (15-20 seconds):**
"Doc2Vec is used to capture the semantic meaning of the entire document. This process involves generating dense vectors that summarize the document as a whole. We fine-tune Doc2Vec by assigning unique identifiers to documents, setting parameters like vector size and window size, and applying mean frequency to balance the representation. The output is a document-level embedding, denoted as 'D,' which provides a global understanding of the text."

---

**Step 3: Fine-Tuning DistilBERT (15-20 seconds):**
"On the other hand, DistilBERT focuses on sentence-level and word-level embeddings. This involves pre-processing the text, tokenizing it, and fine-tuning the model's weights. A multi-head attention mechanism is applied to identify the most relevant tokens, enhancing the focus on critical parts of the sentence. The result is a detailed, contextualized sentence embedding, denoted as 'S.'"

---

**Step 4: Hierarchical Attention Network (HAN) (25-30 seconds):**
"Now, the Hierarchical Attention Network, or HAN, integrates the sentence embeddings from DistilBERT. First, we apply word-level attention to determine the importance of individual words within each sentence. Then, we apply sentence-level attention to evaluate the significance of each sentence in the document. These steps create a context vector, 'v,' which combines the most relevant information from the text. Essentially, HAN ensures that the model focuses on the critical words and sentences, improving interpretability and performance."

---

**Step 5: Combining Embeddings (10-15 seconds):**
"Finally, we concatenate the document-level embeddings from Doc2Vec and the sentence-level embeddings from DistilBERT to form a unified representation, denoted as '(D, S).' This hybrid embedding captures both the global context of the document and the fine-grained details at the sentence level."

---

**Conclusion (10-15 seconds):**
"To summarize, this architecture leverages the strengths of Doc2Vec and DistilBERT, combining them with the Hierarchical Attention Network to create a comprehensive model. By capturing both global and local semantic features, this hybrid approach significantly enhances the accuracy and interpretability of software requirements classification."

---

**Transition (Optional):**
"Now that we've understood the architecture, let's move on to how this model performs and its applications."

---
