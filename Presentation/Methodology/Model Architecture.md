The provided figure represents the **architecture of the H2AN-BiLSTM model**, detailing the processes involved in combining **document-level embeddings (Doc2Vec)** and **sentence-level embeddings (DistilBERT)** using a **Hierarchical Attention Network (HAN)**.

### **Key Sections of the Figure**

#### 1. **Input Data (Labeled Dataset)**
- The dataset (`reviews.csv`) consists of labeled user reviews.
- These reviews are used as input for both **Doc2Vec** and **DistilBERT pipelines** to generate embeddings.

---

#### 2. **Fine-Tuning DistilBERT**
- **Pre-process Text**: The reviews are tokenized and pre-processed to standardize input for the DistilBERT model.
- **Set Parameters**: Parameters such as learning rate, batch size, and epochs are configured.
- **Fine-tune Weights**: The pre-trained DistilBERT model is fine-tuned on the labeled dataset.
- **Multi-Head Attention Mechanism**: This mechanism highlights the most relevant tokens in the sequence, enhancing the focus on critical words.
- **Output**: The output of this process is a **sentence-level embedding (S)**, which captures detailed, contextualized information about the text.

---

#### 3. **Fine-Tuning Doc2Vec**
- **Generate Dense Vector**: Doc2Vec generates dense vectors representing the semantic content of an entire document.
- **Assign Unique Identifiers**: Each document is assigned a unique identifier for better indexing.
- **Set Parameters**: Hyperparameters such as vector size and window size are defined.
- **Apply Mean Frequency**: Word frequencies are considered to generate a balanced representation of text.
- **Output**: The output is a **document-level embedding (D)**, representing the global semantic context of the entire document.

---

#### 4. **Hierarchical Attention Network (HAN)**
- HAN is the core component of the architecture that combines the embeddings from **DistilBERT** and **Doc2Vec**.
- **Input Sequence**: Sentence embeddings from DistilBERT are fed into the HAN.
- **Attention Mechanism**:
  - **Word-Level Attention**: Determines the importance of each word in a sentence. Outputs word vectors (`w_t`), weighted by their relevance.
  - **Sentence-Level Attention**: Evaluates the significance of each sentence in the document. Outputs sentence vectors (`e_u`), weighted by sentence scores (`u_i`).
  - **Context Vector**: Combines these weighted embeddings into a single vector (`v`), representing the most salient information in the text.
- This process ensures the model focuses on the most important words and sentences in the text.

---

#### 5. **Embedding Combination**
- The outputs from **Doc2Vec (D)** and **DistilBERT (S)** are concatenated to form a **unified embedding (D, S)**.
- This combined embedding captures both:
  - **Global semantics** (from Doc2Vec).
  - **Local context** (from DistilBERT and HAN).

---

### **Significance of Each Component**
- **Doc2Vec**: Captures the overall theme and structure of the document, ensuring a broad understanding.
- **DistilBERT**: Provides deep, contextualized insights at the word and sentence level, offering detailed granularity.
- **Hierarchical Attention Network (HAN)**: Optimally combines these embeddings by emphasizing the most relevant parts of the document, ensuring interpretability and improved performance.
- **Combined Embedding**: Bridges the gap between document-level and sentence-level representations, making the model versatile and powerful for software requirements classification.

---
