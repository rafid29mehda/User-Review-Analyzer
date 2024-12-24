# Presentation Script for "H2AN-BiLSTM: A Hierarchical Attention Model for Classifying Software Requirements"

---

### Slide 1: Title Slide
"Good [morning/afternoon], everyone. My name is [Your Name], and today I’ll be presenting our research titled *H2AN-BiLSTM: A Hierarchical Attention Model for Classifying Software Requirements*. This work addresses challenges in software requirements classification and introduces a novel hybrid model for achieving higher accuracy."

---

### Slide 2: Introduction
"To begin, software requirements, or SRs, form the backbone of any software project. Accurate classification of SRs into functional requirements (FRs) and non-functional requirements (NFRs) is critical. This classification ensures better project prioritization, optimal resource allocation, and robust system functionality. However, misclassification can result in costly design flaws and inefficiencies. Traditional methods for classification, such as manual approaches, are time-consuming and prone to error, motivating the need for automated solutions."

---

### Slide 3: Motivation
"Although methods like rule-based approaches, machine learning, and even deep learning have been applied to SR classification, each has limitations. Document-level models like Doc2Vec lack granularity, while word-level models such as BERT fail to capture the broader context. This gap highlights the need for a hybrid approach that integrates global document structure with detailed word-level semantics."

---

### Slide 4: Proposed Model Overview
"Our solution, the H2AN-BiLSTM model, bridges this gap. It stands for Hierarchical Attention Network with Bidirectional LSTM. The key objective is to enhance SR classification by combining document-level embeddings from Doc2Vec with contextualized word-level embeddings from DistilBERT. The model employs a hierarchical attention mechanism to prioritize significant words and sentences and uses a Bidirectional LSTM to capture sequential dependencies, offering a comprehensive understanding of the text."

---

### Slide 5: Datasets Used
"To train and evaluate our model, we utilized two datasets:

1. The PROMISE dataset, which includes 969 labeled requirements—444 functional and 525 non-functional.
2. A larger, unlabeled dataset of 12,495 user reviews from Kaggle, which we annotated using a fine-tuned model trained on PROMISE.

This combination allowed us to expand our analysis to more diverse data while maintaining high accuracy."

---

### Slide 6: Methodology - Preprocessing
"Data preprocessing involved several steps. First, we cleaned the text to remove noise, then tokenized the data using BERT’s tokenizer, converting text into numerical tokens. Functional requirements were labeled as ‘0’ and non-functional as ‘1’ for binary classification. To standardize input, we applied padding and truncation. We also transformed the data into PyTorch tensors for efficient training."

---

### Slide 7: Proposed Model Architecture
"The H2AN-BiLSTM architecture combines several components:

1. Doc2Vec for document-level embeddings, capturing the overall semantic structure.
2. DistilBERT for word-level embeddings, providing detailed contextual information.
3. A Hierarchical Attention Network that focuses on the most important words and sentences.
4. A Bidirectional LSTM, which processes sequences in both forward and backward directions to enhance contextual understanding."

---

### Slide 8: Hierarchical Attention Mechanism
"The attention mechanism plays a central role in our model. It calculates attention scores for each word and sentence, normalizes them, and uses these scores to create a context vector that highlights the most critical parts of the text. This ensures that the model captures both global and local features effectively, improving classification accuracy."

---

### Slide 9: Training and Implementation
"We trained the model on a system with an Intel Core i3 processor and 4GB of RAM. The training process included up to 50 epochs with early stopping to prevent overfitting. Cross-entropy loss optimization was used, and hyperparameters like learning rate and batch size were tuned using grid search. We split the data into 80% training and 20% validation and testing for robust evaluation."

---

### Slide 10: Results and Performance
"Now, let’s discuss the results. Our model achieved high precision, recall, and F1-scores across the training, validation, and test datasets. On the test set, the accuracy was 94.40%, with precision and recall consistently above 92% for both FRs and NFRs. This demonstrates the robustness and generalizability of our approach."

---

### Slide 11: Comparative Analysis
"When compared to other models, such as Bag-of-Words, Word2Vec, and standalone BERT or DistilBERT models, our hybrid H2AN-BiLSTM significantly outperforms them. For instance, the accuracy of BERT was 91%, while our model achieved 94.40%, showcasing the effectiveness of combining document- and word-level embeddings."

---

### Slide 12: Strengths of the Model
"The strengths of our model lie in its ability to:

1. Integrate document-level and word-level semantics for a holistic understanding of the text.
2. Capture long-range dependencies using Bi-LSTM.
3. Enhance interpretability through hierarchical attention mechanisms.
4. Handle large datasets efficiently while maintaining high accuracy."

---

### Slide 13: Limitations and Future Work
"While our model achieves impressive results, it has some limitations. It focuses only on binary classification of FRs and NFRs and relies heavily on pre-trained embeddings. Future work could extend the model to multi-label classification, apply it to other domains such as healthcare or legal requirements, and explore domain-specific pre-trained models for further improvement."

---

### Slide 14: Conclusion
"In conclusion, the H2AN-BiLSTM model provides a novel approach to classifying software requirements. By combining document-level and word-level embeddings, it achieves state-of-the-art accuracy of 94.40%. This hybrid solution is robust, scalable, and significantly improves the automation of requirement classification, offering a valuable tool for software engineers."

---

### Slide 15: Thank You
"Thank you for your attention. I am happy to take any questions you may have. Feel free to connect with me via email or LinkedIn if you would like to discuss this further."

---

