**Framework of the Proposed Approach**

---

To effectively illustrate the methodology and workflow of our proposed hybrid NLP model, we present a detailed framework outlining each component and their interactions. This framework serves as a roadmap of the processes involved, from data preparation to the final classification using combined embeddings. Below is a comprehensive description of the framework.

### **Overview**

The proposed approach consists of the following major components:

1. **Data Acquisition and Preparation**
   - Collection of datasets (`PROMISE_exp.csv` and `reviews.csv`).
   - Labeling the unlabeled reviews using a fine-tuned BERT model.

2. **Model Training**
   - Fine-tuning the Doc2Vec model on the labeled dataset.
   - Fine-tuning the DistilBERT model on the same dataset.

3. **Embedding Extraction**
   - Generating document-level embeddings using Doc2Vec.
   - Generating contextualized word-level embeddings using DistilBERT.

4. **Embedding Integration and Classification**
   - Concatenating Doc2Vec and DistilBERT embeddings.
   - Training an XGBoost classifier on the combined embeddings.
   - Evaluating the model's performance.

### **Detailed Framework Components**

#### **1. Data Acquisition and Preparation**

**1.1. Collect Datasets**

- **PROMISE_exp.csv**: A publicly available dataset containing requirements labeled as Functional (F) or Non-Functional (NF).
- **reviews.csv**: An unlabeled dataset of user reviews obtained from the Hugging Face dataset collection.

**1.2. Labeling Unlabeled Reviews**

- **Fine-tune BERT on PROMISE_exp.csv**:
  - Map labels 'F' and 'NF' to integers (F → 0, NF → 1).
  - Split the data into training and validation sets.
  - Use the Hugging Face `Trainer` to fine-tune the BERT model for sequence classification.

- **Classify `reviews.csv`**:
  - Use the fine-tuned BERT model to predict labels for each review in `reviews.csv`.
  - Assign labels to the reviews, creating `labeled_reviews.csv`.

**1.3. Data Preprocessing**

- Clean and preprocess text data (e.g., lowercasing, tokenization).
- Ensure consistency in data formats across models.

#### **2. Model Training**

**2.1. Fine-tuning Doc2Vec**

- **Prepare Tagged Documents**:
  - Tokenize the text of each review.
  - Create `TaggedDocument` objects with tokens and unique tags.

- **Initialize and Train Doc2Vec Model**:
  - Set hyperparameters (e.g., `vector_size=100`, `window=5`, `epochs=20`).
  - Build the vocabulary from the tagged documents.
  - Train the model on the labeled dataset.

- **Save the Trained Model**:
  - Save the fine-tuned Doc2Vec model for later use in embedding extraction.

**2.2. Fine-tuning DistilBERT**

- **Prepare Dataset**:
  - Split the labeled data into training and test sets.
  - Tokenize the text using the DistilBERT tokenizer with appropriate padding and truncation.

- **Initialize and Train DistilBERT Model**:
  - Use the `DistilBertForSequenceClassification` model with `num_labels=2`.
  - Set training arguments (e.g., `learning_rate=2e-5`, `num_train_epochs=3`).
  - Fine-tune the model using the Hugging Face `Trainer`.

- **Save the Trained Model**:
  - Save the fine-tuned DistilBERT model and tokenizer for embedding extraction.

#### **3. Embedding Extraction**

**3.1. Generate Doc2Vec Embeddings**

- **Load Trained Doc2Vec Model**:
  - Load the saved Doc2Vec model from disk.

- **Extract Embeddings**:
  - For each review, retrieve the corresponding document vector using the document's tag.

**3.2. Generate DistilBERT Embeddings**

- **Load Trained DistilBERT Model and Tokenizer**:
  - Load the fine-tuned DistilBERT model and tokenizer.

- **Extract Embeddings**:
  - For each review:
    - Tokenize the text.
    - Pass the tokens through the DistilBERT model to obtain the last hidden states.
    - Apply average pooling over the token embeddings to get a fixed-size vector representation.

#### **4. Embedding Integration and Classification**

**4.1. Concatenate Embeddings**

- **Combine Embeddings**:
  - For each review, concatenate the Doc2Vec embedding (document-level) and the DistilBERT embedding (contextualized word-level) to form a single feature vector.

**4.2. Prepare Data for Classification**

- **Feature Matrix (`X`)**:
  - Compile all combined embeddings into a feature matrix.

- **Target Vector (`y`)**:
  - Use the labels assigned during data preparation.

- **Data Splitting**:
  - Split the data into training, validation, and test sets (e.g., 70% training, 15% validation, 15% test).

**4.3. Train XGBoost Classifier**

- **Initialize Classifier**:
  - Set hyperparameters (e.g., `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`).

- **Train Classifier**:
  - Fit the XGBoost classifier on the training data.

**4.4. Evaluate Model Performance**

- **Make Predictions**:
  - Use the trained classifier to predict labels on the validation and test sets.

- **Compute Evaluation Metrics**:
  - Calculate accuracy, precision, recall, F1-score, and confusion matrices for each dataset split.

- **Analyze Results**:
  - Compare the performance metrics to assess the effectiveness of the hybrid model.

### **Framework Flowchart Description**

While a visual diagram cannot be rendered here, the following description outlines the flow of the framework:

1. **Data Input**:
   - **PROMISE_exp.csv** → Fine-tune BERT → Label `reviews.csv`
   - **reviews.csv** → Labeled as `labeled_reviews.csv`

2. **Model Fine-tuning**:
   - **Doc2Vec**:
     - Input: `labeled_reviews.csv`
     - Process: Tokenization, Tagging, Model Training
     - Output: Trained Doc2Vec Model
   - **DistilBERT**:
     - Input: `labeled_reviews.csv`
     - Process: Tokenization, Model Training
     - Output: Trained DistilBERT Model

3. **Embedding Extraction**:
   - **Doc2Vec Embeddings**:
     - Use the trained Doc2Vec model to generate embeddings for each review.
   - **DistilBERT Embeddings**:
     - Use the trained DistilBERT model to generate embeddings for each review.

4. **Embedding Integration**:
   - Concatenate Doc2Vec and DistilBERT embeddings for each review to form combined embeddings.

5. **Classification**:
   - **Input**: Combined embeddings (`X`), Labels (`y`)
   - **Process**:
     - Split data into training, validation, and test sets.
     - Train XGBoost classifier on training data.
     - Validate and test the classifier.
   - **Output**: Predictions, Evaluation Metrics

6. **Model Evaluation**:
   - Analyze the classifier's performance using metrics and confusion matrices.
   - Compare results with individual models.

### **Key Advantages of the Framework**

- **Comprehensive Feature Representation**:
  - By combining embeddings from Doc2Vec and DistilBERT, the model captures both global document semantics and local contextual nuances.

- **Robust Classification**:
  - The use of XGBoost leverages its strength in handling structured data and complex feature interactions, improving classification performance.

- **Scalability**:
  - The framework can be extended to other text classification tasks by replacing or augmenting the datasets and models.

### **Implementation Considerations**

- **Computational Resources**:
  - Training large models like DistilBERT and combining high-dimensional embeddings require substantial computational power.

- **Data Quality**:
  - The effectiveness of the model depends on the quality and representativeness of the labeled dataset.

- **Hyperparameter Tuning**:
  - Optimal performance may require tuning hyperparameters for Doc2Vec, DistilBERT, and XGBoost.

### **Conclusion**

The proposed framework provides a systematic approach to classify user reviews by integrating different levels of semantic information. This hybrid methodology demonstrates enhanced performance over individual models, validating the effectiveness of combining multiple embedding techniques in NLP classification tasks.

---

