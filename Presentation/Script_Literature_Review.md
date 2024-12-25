### Script to Explain the Literature Review Using the Table

---

**[Start]**

"Now, let’s dive into the Literature Review. This section provides an overview of related works and identifies the research gaps that our paper aims to address. I’ll walk you through the key contributions and limitations of previous studies using the table."

---

**[Point to Table]**

"First, let’s start with Lima et al. (2019). They focused on expanding the PROMISE database by incorporating new data and applied machine learning techniques such as Decision Tree and Naïve Bayes for software requirements classification. While this improved accuracy, the manual nature of the expansion process led to significant class imbalance in the dataset, which limited its robustness."

"Next, Younas et al. (2019) proposed a semantic similarity-based method for extracting non-functional requirements using Word2Vec embeddings. While they achieved an F1-score of 64%, their approach struggled with accuracy and failed to capture nuanced semantic relationships, which are crucial for precise classification."

---

**[Transition to Advanced Models]**

"Moving on to more advanced techniques, Tasnim et al. (2023) introduced an attention-based LSTM model. This approach emphasized critical portions of the text, achieving an impressive accuracy of 94.20%. However, their work was constrained to small datasets, making it less generalizable to larger, diverse datasets like those used in real-world applications."

"Kici et al. (2021) applied a transformer-based approach using DistilBERT. By fine-tuning the model, they achieved an F1-score of 80.79% and suggested exploring domain-specific pre-trained models. Despite its promise, this approach underperformed compared to hybrid or more comprehensive models, as it lacked document-level understanding."

---

**[Highlight Hybrid and Comparative Models]**

"Rahimi et al. (2020) took an ensemble approach, combining LSTM, GRU, CNN, and BiLSTM to capture both short- and long-term dependencies. While this effectively classified functional requirements, the study primarily focused on functional requirements, overlooking non-functional or quality attributes that are just as important."

"Finally, Cruciani et al. (2023) compared multiple general-purpose embeddings such as BERT, DistilBERT, SBERT, and GloVe. Their evaluation highlighted that BERT achieved the highest F1-scores of 90.36%. However, their work lacked the integration of document-level and word-level embeddings and didn’t utilize sequential modeling capabilities."

---


This script provides a structured explanation while transitioning smoothly into the need for your proposed model. 
