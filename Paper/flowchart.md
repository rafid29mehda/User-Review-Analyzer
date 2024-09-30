**Flowchart of the Proposed Hybrid NLP Model Framework**

---

---

### **Flowchart Description**

#### **Start**

- **Begin**: Start of the flowchart.

#### **1. Data Acquisition and Preparation**

**1.1. Collect Datasets**

- **Input Node**: *PROMISE_exp.csv* (Labeled Requirements Dataset)
- **Input Node**: *reviews.csv* (Unlabeled User Reviews Dataset)

**1.2. Labeling Unlabeled Reviews**

- **Process Block**: Fine-tune BERT on *PROMISE_exp.csv*
  - **Sub-process**:
    - Map labels 'F' and 'NF' to integers.
    - Split into training and validation sets.
    - Fine-tune BERT model for sequence classification.
- **Process Block**: Use fine-tuned BERT to label *reviews.csv*
  - **Output**: *labeled_reviews.csv* (Labeled User Reviews Dataset)

**Decision Point**: Is the labeling of *reviews.csv* complete?
- **Yes**: Proceed to next step.
- **No**: Return to labeling process.

#### **2. Model Training**

**2.1. Fine-tuning Doc2Vec**

- **Process Block**: Train Doc2Vec Model
  - **Input**: *labeled_reviews.csv*
  - **Sub-process**:
    - Tokenize text data.
    - Create `TaggedDocument` objects.
    - Initialize Doc2Vec model with hyperparameters.
    - Build vocabulary and train the model.
  - **Output**: Trained Doc2Vec Model

**2.2. Fine-tuning DistilBERT**

- **Process Block**: Train DistilBERT Model
  - **Input**: *labeled_reviews.csv*
  - **Sub-process**:
    - Tokenize text data using DistilBERT tokenizer.
    - Split data into training and test sets.
    - Fine-tune DistilBERT model for sequence classification.
  - **Output**: Trained DistilBERT Model

#### **3. Embedding Extraction**

**3.1. Generate Doc2Vec Embeddings**

- **Process Block**: Extract Doc2Vec Embeddings
  - **Input**: Trained Doc2Vec Model, *labeled_reviews.csv*
  - **Process**:
    - For each review, obtain the document vector.
  - **Output**: Doc2Vec Embeddings

**3.2. Generate DistilBERT Embeddings**

- **Process Block**: Extract DistilBERT Embeddings
  - **Input**: Trained DistilBERT Model, *labeled_reviews.csv*
  - **Process**:
    - For each review, tokenize text.
    - Pass tokens through DistilBERT to get last hidden states.
    - Apply average pooling to obtain fixed-size embeddings.
  - **Output**: DistilBERT Embeddings

#### **4. Embedding Integration and Classification**

**4.1. Concatenate Embeddings**

- **Process Block**: Combine Embeddings
  - **Input**: Doc2Vec Embeddings, DistilBERT Embeddings
  - **Process**:
    - Concatenate the two embeddings for each review.
  - **Output**: Combined Embeddings (Feature Matrix `X`)

**4.2. Prepare Data for Classification**

- **Process Block**: Prepare Dataset
  - **Input**: Combined Embeddings, Labels (`y`)
  - **Process**:
    - Split data into Training, Validation, and Test sets.
  - **Output**: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`

**4.3. Train XGBoost Classifier**

- **Process Block**: Train Classifier
  - **Input**: `X_train`, `y_train`
  - **Process**:
    - Initialize XGBoost classifier with hyperparameters.
    - Train classifier on training data.
  - **Output**: Trained XGBoost Model

**4.4. Evaluate Model Performance**

- **Process Block**: Evaluate Model
  - **Input**: Trained XGBoost Model, `X_val`, `y_val`, `X_test`, `y_test`
  - **Process**:
    - Predict labels on validation and test sets.
    - Compute evaluation metrics (accuracy, precision, recall, F1-score).
    - Generate confusion matrices.
  - **Decision Point**: Is the performance satisfactory?
    - **Yes**: Proceed to conclusion.
    - **No**: Return to hyperparameter tuning or data preprocessing.

#### **End**

- **Output Node**: Final model and evaluation results.
- **End of Flowchart**

---

### **Flowchart Components and Connections**

**1. Start Node**

- Symbol: Rounded rectangle labeled "Start"

**2. Input Nodes**

- Symbols: Parallelograms labeled with the dataset names:
  - "PROMISE_exp.csv"
  - "reviews.csv"

**3. Process Blocks**

- Symbols: Rectangles representing processes, labeled accordingly:
  - "Fine-tune BERT on PROMISE_exp.csv"
  - "Label reviews.csv with fine-tuned BERT"
  - "Train Doc2Vec Model"
  - "Train DistilBERT Model"
  - "Extract Doc2Vec Embeddings"
  - "Extract DistilBERT Embeddings"
  - "Combine Embeddings"
  - "Prepare Dataset for Classification"
  - "Train XGBoost Classifier"
  - "Evaluate Model Performance"

**4. Decision Points**

- Symbols: Diamonds representing decisions, labeled with the condition:
  - "Is labeling complete?"
  - "Is performance satisfactory?"

**5. Data Flow**

- Arrows connecting the components, indicating the flow of data and processes:
  - From Start to Data Acquisition
  - From Input Nodes to Process Blocks
  - Between Process Blocks as per the sequence described
  - From Decision Points to different paths based on "Yes" or "No" outcomes

**6. Output Nodes**

- Symbols: Parallelograms labeled with outputs:
  - "labeled_reviews.csv"
  - "Trained Doc2Vec Model"
  - "Trained DistilBERT Model"
  - "Doc2Vec Embeddings"
  - "DistilBERT Embeddings"
  - "Combined Embeddings (Feature Matrix X)"
  - "Training, Validation, Test Sets"
  - "Trained XGBoost Model"
  - "Evaluation Metrics and Confusion Matrices"
  - "Final Model and Results"

**7. End Node**

- Symbol: Rounded rectangle labeled "End"

---

### **Instructions to Create the Flowchart**

Using the above description, you can create the flowchart by following these steps:

1. **Set Up the Canvas**

   - Start with a blank canvas in your diagramming tool.
   - Decide on the flow direction (top-down or left-to-right).

2. **Add Start and End Nodes**

   - Place a "Start" node at the beginning.
   - Place an "End" node at the conclusion.

3. **Add Input and Output Nodes**

   - Add parallelograms for the datasets at the beginning.
   - Include output nodes at appropriate steps.

4. **Add Process Blocks**

   - Insert rectangles for each major process.
   - Label them clearly with the process descriptions.

5. **Add Decision Points**

   - Use diamonds for decision points.
   - Label them with the condition questions.

6. **Connect the Components**

   - Use arrows to connect the nodes, showing the flow.
   - Indicate the direction of data and process flow.
   - For decision points, label the outgoing arrows with "Yes" or "No".

7. **Annotate if Necessary**

   - Include brief notes or annotations beside certain steps for clarity.
   - For example, specify key hyperparameters or tools used.

8. **Review and Refine**

   - Ensure that all components are connected logically.
   - Check for completeness and accuracy against the framework description.

---

### **Example Flow**

Here is a simplified linear representation of the flowchart steps:

1. **Start**
2. **Collect Datasets**
   - PROMISE_exp.csv
   - reviews.csv
3. **Fine-tune BERT on PROMISE_exp.csv**
4. **Label reviews.csv using fine-tuned BERT**
   - Output: labeled_reviews.csv
5. **Train Doc2Vec Model on labeled_reviews.csv**
   - Output: Trained Doc2Vec Model
6. **Train DistilBERT Model on labeled_reviews.csv**
   - Output: Trained DistilBERT Model
7. **Extract Doc2Vec Embeddings**
   - Input: Trained Doc2Vec Model, labeled_reviews.csv
   - Output: Doc2Vec Embeddings
8. **Extract DistilBERT Embeddings**
   - Input: Trained DistilBERT Model, labeled_reviews.csv
   - Output: DistilBERT Embeddings
9. **Combine Embeddings**
   - Output: Combined Embeddings (Feature Matrix X)
10. **Prepare Data for Classification**
    - Split into training, validation, test sets
11. **Train XGBoost Classifier**
12. **Evaluate Model Performance**
    - If performance satisfactory, proceed to End
    - If not, loop back to previous steps (e.g., hyperparameter tuning)
13. **End**

---

