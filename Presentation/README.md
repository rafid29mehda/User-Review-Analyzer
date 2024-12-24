# Presentation on "H2AN-BiLSTM: A Hierarchical Attention Model for Classifying Software Requirements"

---

### Slide 1: **Title Slide**
- **Title**: H2AN-BiLSTM: A Hierarchical Attention Model for Classifying Software Requirements
- **Presenter Name**: [Your Name]
- **Conference Name**: [Conference Name]
- **Date**: [Presentation Date]

---

### Slide 2: **Introduction**
- **Software Requirements (SRs)** are critical for effective software engineering.
- Proper classification of SRs into Functional Requirements (FRs) and Non-Functional Requirements (NFRs) is essential for:
  - Project prioritization
  - Resource allocation
  - Ensuring system functionality
- **Challenges**:
  - Misclassification of SRs can lead to costly design flaws.
  - Traditional/manual methods are error-prone and inefficient.

---

### Slide 3: **Motivation**
- **Traditional Approaches**:
  - Rule-based/manual methods
  - Machine Learning (ML): SVM, Naïve Bayes, etc.
  - Deep Learning (DL): BERT, LSTM, etc.
- **Limitations**:
  - Document-level models (e.g., Doc2Vec) lack granularity.
  - Word-level models (e.g., BERT) fail to capture broader context.
- **Need**: A hybrid model to integrate global structure and contextual details.

---

### Slide 4: **Proposed Model Overview**
- **Model Name**: H2AN-BiLSTM (Hierarchical Attention Network with Bi-LSTM).
- **Objective**:
  - Improve SR classification accuracy.
  - Combine document-level semantics (Doc2Vec) and word-level embeddings (DistilBERT).
- **Key Features**:
  - Hierarchical Attention Network (HAN): Prioritizes significant words/sentences.
  - Bidirectional LSTM: Captures sequential dependencies.

---

### Slide 5: **Datasets Used**
1. **PROMISE Dataset**:
   - 969 requirements: 444 FRs, 525 NFRs.
2. **Unlabeled Reviews Dataset**:
   - 12,495 user reviews from Kaggle.
   - Annotated using the fine-tuned PROMISE model.

| Dataset         | Functional (FR) | Non-Functional (NFR) | Total  |
|-----------------|----------------|-----------------------|--------|
| PROMISE         | 444            | 525                   | 969    |
| Reviews Dataset | 5,552          | 6,943                 | 12,495 |

---

### Slide 6: **Methodology - Preprocessing**
- **Steps**:
  1. Text cleaning: Removing noise.
  2. Tokenization: Converting text into tokens using BERT tokenizer.
  3. Numerical Mapping: FR (‘0’), NFR (‘1’).
  4. Uniform Input Length: Padding/truncation for batch processing.
- **Tools Used**:
  - PyTorch for tensor transformations.
  - Pre-trained BERT for token embeddings.

---

### Slide 7: **Proposed Model Architecture**
1. **Document-Level Embeddings**:
   - Doc2Vec: Captures semantic content of entire documents.
2. **Word-Level Embeddings**:
   - DistilBERT: Generates 768-dimensional vectors for each token.
3. **Hierarchical Attention Network (HAN)**:
   - Selectively focuses on important words and sentences.
4. **Bidirectional LSTM (Bi-LSTM)**:
   - Processes sequences in both forward and backward directions.
   - Enhances context understanding.

---

### Slide 8: **Hierarchical Attention Mechanism**
- **Attention Formulae**:
  - Attention score: “uᵢ = tanh(Wᵢeᵢ + bᵢ)”
  - Normalized weight: “αᵢ = exp(uᵢ) / Σexp(uᵣ)”
  - Context vector: “v = Σαᵢeᵢ”
- **Benefits**:
  - Identifies key parts of text.
  - Balances global and local features.

---

### Slide 9: **Training and Implementation**
- **Setup**:
  - System: Intel Core i3, 4GB RAM.
  - Tools: Python, Jupyter Notebook.
- **Training Details**:
  - 50 epochs with early stopping.
  - Cross-entropy loss optimization.
  - Grid search for hyperparameter tuning.
- **Train-Test Split**:
  - 80% Training, 20% Validation/Test.

---

### Slide 10: **Results and Performance**
- **Metrics Evaluated**:
  - Precision, Recall, F1-score, Accuracy.
- **Key Results**:
  | Dataset         | Precision (%) | Recall (%) | F1-Score (%) | Accuracy (%) |
  |-----------------|---------------|------------|--------------|--------------|
  | Training        | 94-97         | 95-96      | 95-96        | 95.88        |
  | Validation      | 94-97         | 95-97      | 95-96        | 95.30        |
  | Test            | 92-97         | 93-96      | 94-95        | 94.40        |

---

### Slide 11: **Comparative Analysis**
| Model            | Accuracy (%) |
|------------------|--------------|
| Bag-of-Words     | 92           |
| Word2Vec         | 87           |
| BERT             | 91           |
| DistilBERT       | 90           |
| H2AN-BiLSTM      | **94.40**    |

- **Observations**:
  - Hybrid model outperforms standalone approaches.
  - Combination of Doc2Vec and DistilBERT is highly effective.

---

### Slide 12: **Strengths of the Model**
- Captures both document-level and word-level semantics.
- Handles long-range dependencies with Bi-LSTM.
- Enhanced interpretability with hierarchical attention.
- High accuracy and scalability for large datasets.

---

### Slide 13: **Limitations and Future Work**
- **Limitations**:
  - Focus on binary classification (FR vs. NFR).
  - Dependence on pre-trained embeddings.
- **Future Directions**:
  - Extend to multi-label classification.
  - Apply to other domains (e.g., medical, legal).
  - Explore domain-specific pre-trained models.

---

### Slide 14: **Conclusion**
- **Summary**:
  - H2AN-BiLSTM effectively classifies FRs and NFRs.
  - Combines Doc2Vec and DistilBERT embeddings for a hybrid solution.
  - Achieves state-of-the-art accuracy (94.40%).
- **Impact**:
  - Provides a scalable, robust tool for software engineers.
  - Automates requirement classification with high precision.

---

### Slide 15: **Thank You**
- **Contact Information**:
  - Email: [Your Email Address]
  - LinkedIn: [Your LinkedIn Profile]
- **Questions?**

