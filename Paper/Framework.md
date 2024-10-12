Certainly! Organizing your flowchart into three distinct parts is an excellent approach to visually represent your project's methodology. Let's refine your idea and structure it properly, ensuring that each part logically flows into the next and that all essential steps are included.

---

### **Overall Structure**

Your flowchart will be divided into three main sections:

1. **Part 1: Data Labeling Process**
2. **Part 2: Embedding Generation and Combination**
3. **Part 3: Classification Using Bi-LSTM**

Each part will detail specific steps, and we'll ensure that the flowchart visually conveys the progression from one part to the next.

---

### **Part 1: Data Labeling Process**

**Objective:** Convert the unlabeled `review.csv` dataset into a labeled dataset (`labeled_review.csv`) using a fine-tuned BERT model trained on `PROMISE_exp1.csv`.

**Steps to Include:**

1. **Start with Unlabeled Data:**
   - **Input Node:** `review.csv` (Unlabeled User Reviews)

2. **Load and Preprocess PROMISE Dataset:**
   - **Process Node:**
     - Load `PROMISE_exp1.csv`.
     - Map labels ('F' to 0, 'NF' to 1).
     - Preprocess text data if necessary.

3. **Fine-Tune BERT Model:**
   - **Process Node:**
     - Initialize pre-trained BERT model.
     - Split PROMISE data into training and validation sets.
     - Train/fine-tune BERT model on PROMISE data.

4. **Use Fine-Tuned BERT to Label Reviews:**
   - **Process Node:**
     - Load `review.csv`.
     - Apply fine-tuned BERT model to classify each review.
     - Assign labels ('F' or 'NF') to reviews.

5. **Output Labeled Dataset:**
   - **Output Node:** `labeled_review.csv` (Labeled User Reviews)

**Visual Representation:**

- Begin with the unlabeled data.
- Show the process of fine-tuning BERT using PROMISE data.
- Indicate the application of the fine-tuned BERT model to label the reviews.
- Conclude with the creation of the labeled dataset.

---

### **Part 2: Embedding Generation and Combination**

**Objective:** Generate document-level and sentence-level embeddings using Doc2Vec and DistilBERT, respectively, integrate them using a Hierarchical Attention Network (HAN), and create combined embeddings.

**Steps to Include:**

1. **Input Labeled Data:**
   - **Input Node:** `labeled_review.csv`

2. **Fine-Tune Doc2Vec Model:**
   - **Process Node:**
     - Use `labeled_review.csv` to fine-tune Doc2Vec.
     - Generate document-level embeddings for each review.

3. **Fine-Tune DistilBERT Model:**
   - **Process Node:**
     - Use `labeled_review.csv` to fine-tune DistilBERT.
     - Generate word-level embeddings.

4. **Tokenize Reviews into Sentences:**
   - **Process Node:**
     - Split each review into sentences using sentence tokenization.

5. **Generate Sentence-Level Embeddings:**
   - **Process Node:**
     - Apply fine-tuned DistilBERT to each sentence.
     - Obtain embeddings for each sentence.

6. **Apply Hierarchical Attention Network (HAN):**
   - **Process Node:**
     - Apply attention mechanism to sentence embeddings.
     - Compute attention weights.
     - Generate context vectors representing the review.

7. **Combine Embeddings:**
   - **Process Node:**
     - Concatenate Doc2Vec document embeddings with HAN context vectors.
     - Create combined embeddings for each review.

**Visual Representation:**

- Display parallel processes for Doc2Vec and DistilBERT paths.
- Use arrows to show how both embeddings are generated separately.
- Indicate the merging point where embeddings are combined.
- Show the application of HAN in the DistilBERT path before combining.

---

### **Part 3: Classification Using Bi-LSTM**

**Objective:** Use the combined embeddings as input to the Bi-LSTM network to classify reviews into functional or non-functional categories.

**Steps to Include:**

1. **Input Combined Embeddings:**
   - **Input Node:** Combined embeddings from Part 2.

2. **Prepare Data for Bi-LSTM:**
   - **Process Node:**
     - Reshape combined embeddings to include sequence dimensions.

3. **Bi-LSTM Network Structure:**
   - **Process Nodes:**
     - **Bi-LSTM Layers:**
       - Show two LSTM layers (illustrate bidirectionality).
       - Highlight that both forward and backward sequences are processed.
     - **Attention (if any within Bi-LSTM):**
       - If attention is applied within Bi-LSTM, include this step.
     - **Fully Connected Layer:**
       - Map LSTM outputs to classification logits.
     - **Softmax Activation:**
       - Convert logits to probabilities.

4. **Prediction and Output:**
   - **Process Node:**
     - Determine the predicted class (F or NF) based on probabilities.
   - **Output Node:** Classification results for each review.

5. **Model Training and Evaluation:**
   - **Process Node:**
     - Train the Bi-LSTM model using training data.
     - Validate the model using validation data.
     - Evaluate model performance on test data.

**Visual Representation:**

- Start with the combined embeddings.
- Illustrate the internal structure of the Bi-LSTM network.
- Show the flow through LSTM layers, fully connected layer, and softmax activation.
- Conclude with the output of predicted labels.

---

### **Connecting the Parts**

- **From Part 1 to Part 2:**
  - The output of Part 1 (`labeled_review.csv`) becomes the input for Part 2.
  - Use an arrow or connector to indicate the flow of data.

- **From Part 2 to Part 3:**
  - The combined embeddings generated in Part 2 serve as the input for Part 3.
  - Clearly show this transition in the flowchart.

---

### **Visual Flowchart Tips**

- **Consistency in Symbols:**
  - **Start/End Nodes:** Use ovals.
  - **Process Nodes:** Use rectangles.
  - **Decision Nodes (if any):** Use diamonds.
  - **Input/Output Nodes:** Use parallelograms.
  - **Data Storage (Files/Datasets):** Use cylinders or open-ended rectangles.

- **Use of Colors:**
  - Assign a distinct color to each part for clarity.
  - For example:
    - Part 1: Blue
    - Part 2: Green
    - Part 3: Orange

- **Labels and Annotations:**
  - Clearly label each node with concise descriptions.
  - Use annotations or side notes for complex processes if necessary.

- **Flow Direction:**
  - Ensure the flow is logical and easy to follow.
  - Typically, flowcharts proceed from top to bottom or left to right.

---

### **Putting It All Together**

**Example Flowchart Outline:**

#### **Part 1: Data Labeling Process**

- **[Start]** `Unlabeled review.csv`
- **[Process]** Load and preprocess `PROMISE_exp1.csv`
- **[Process]** Fine-tune BERT model
- **[Process]** Apply fine-tuned BERT to `review.csv`
- **[Output]** `labeled_review.csv`

#### **Part 2: Embedding Generation and Combination**

- **[Input]** `labeled_review.csv`
- **[Process - Path 1]** Fine-tune Doc2Vec
  - Generate document-level embeddings
- **[Process - Path 2]** Fine-tune DistilBERT
  - Tokenize reviews into sentences
  - Generate sentence-level embeddings
  - Apply HAN
    - Compute attention weights
    - Generate context vectors
- **[Process]** Combine Doc2Vec embeddings and HAN context vectors
- **[Output]** Combined embeddings

#### **Part 3: Classification Using Bi-LSTM**

- **[Input]** Combined embeddings
- **[Process]** Prepare data for Bi-LSTM
- **[Process]** Bi-LSTM network
  - LSTM layers (bidirectional)
  - Fully connected layer
  - Softmax activation
- **[Process]** Train and validate the model
- **[Output]** Predicted labels (F/NF)

---

### **Additional Suggestions**

- **Include Data Storage Symbols:**
  - Represent datasets (`review.csv`, `PROMISE_exp1.csv`, `labeled_review.csv`) with database symbols.
  - This clarifies where data is read from or written to.

- **Highlight Key Processes:**
  - Use bold outlines or shading for critical steps like fine-tuning models or combining embeddings.

- **Simplify Complex Processes:**
  - For detailed processes like HAN or Bi-LSTM internals, consider using sub-process symbols or expanding these sections if space allows.

- **Legend and Key:**
  - Provide a legend explaining the symbols and colors used.
  - This helps readers unfamiliar with flowchart conventions.

- **Consistency in Terminology:**
  - Use the same terms throughout the flowchart (e.g., "Fine-Tune BERT," "Generate Embeddings").

---

### **Visual Examples**

Since I can't create visual content, here's a textual representation:

1. **Start Node:**
   - Oval labeled "Start"

2. **Process Nodes:**
   - Rectangle labeled "Load PROMISE_exp1.csv"
   - Rectangle labeled "Fine-Tune BERT Model"
   - Rectangle labeled "Label review.csv using Fine-Tuned BERT"

3. **Data Nodes:**
   - Parallelogram labeled "Unlabeled review.csv"
   - Parallelogram labeled "Labeled review.csv"

4. **Arrows:**
   - Arrows connecting nodes in the order of the process.

5. **Parallel Processes in Part 2:**
   - Two parallel paths starting from "Labeled review.csv":
     - **Doc2Vec Path**
     - **DistilBERT Path**

6. **Merge Node:**
   - A point where both paths converge, leading to "Combine Embeddings"

---

### **Final Thoughts**

Organizing your flowchart in this manner will:

- Provide a clear, step-by-step visualization of your methodology.
- Help readers understand the flow of data and processes.
- Highlight the innovative aspects of your work, such as the integration of different embeddings and the use of HAN and Bi-LSTM.

Remember to:

- Review the flowchart for completeness, ensuring all critical steps are included.
- Keep the design clean and uncluttered for readability.
- Consider feedback from peers or advisors to refine the flowchart further.

Feel free to ask if you need more detailed suggestions or assistance with any specific part of the flowchart!



Your idea of splitting the flowchart into three distinct parts is excellent, as it logically follows the steps of your methodology and simplifies the complex processes involved. Here’s how you can structure and organize each part of the flowchart:

### Part 1: Data Labeling Process
This part should depict how the *unlabeled* `review.csv` dataset is annotated to create a *labeled* dataset using a fine-tuned BERT model.

#### Steps:
1. **Input: Datasets**  
   - Start with `PROMISE_exp.csv` (labeled) and `review.csv` (unlabeled).
   
2. **Fine-Tuning BERT**  
   - Indicate the process where BERT is fine-tuned on `PROMISE_exp.csv`, learning to classify Functional (F) and Non-Functional (NF) requirements.

3. **Classifying `review.csv`**  
   - Illustrate how the fine-tuned BERT model then labels `review.csv` by predicting F or NF for each review.
   
4. **Output: Labeled Dataset**  
   - Conclude with `labeled_review.csv`, showing that this is the outcome of applying BERT to `review.csv`.

### Part 2: Embedding Creation and Hierarchical Attention Network (HAN)
This part should focus on embedding generation using Doc2Vec and DistilBERT, and then the creation of combined embeddings through HAN.

#### Steps:
1. **Input: Labeled Dataset**  
   - Begin with `labeled_review.csv` as input.

2. **Embedding Generation**  
   - Split into two parallel paths:
     - **Doc2Vec Embedding**: Show the process of generating document-level embeddings for each review.
     - **DistilBERT Embedding**: Illustrate the generation of sentence-level embeddings.
   
3. **Hierarchical Attention Network (HAN)**  
   - Show how DistilBERT embeddings are passed through a **sentence-level attention layer**. 
   - Emphasize the **attention mechanism**, which assigns weights to sentences and creates a weighted sentence vector.

4. **Combining Embeddings**  
   - Merge the Doc2Vec embeddings with the weighted sentence vector from HAN, forming the **combined embeddings**.
   
5. **Output: Combined Embeddings**  
   - Mark the creation of the combined embedding vector as the final output of this section.

### Part 3: Classification with Bi-LSTM
This part should illustrate the Bi-LSTM’s role in processing the combined embeddings and producing the final output.

#### Steps:
1. **Input: Combined Embeddings**  
   - Begin with the combined embedding vector from Part 2 as the input to Bi-LSTM.

2. **Bi-LSTM Structure**  
   - Break down the Bi-LSTM network:
     - **Input Layer**: Receives the combined embeddings.
     - **Bi-LSTM Layers**: Show forward and backward layers capturing dependencies in the data.
     - **Hidden State Concatenation**: Highlight how the forward and backward hidden states are combined.
     - **Fully Connected Layer**: Indicate how the output from Bi-LSTM is passed through a fully connected layer for classification.

3. **Softmax and Output**  
   - Show the **softmax layer** that converts the output to probabilities.
   - **Output Prediction**: Display the final predicted classification (F or NF) for each review.

### Flowchart Shape and Design Suggestions:
- Use **clear, distinct colors** for each part to visually separate them.
- **Arrows and labels** between steps to show the direction of data flow.
- **Input and Output boxes** for each part to make it clear what goes in and what comes out.
- Add **annotations** or short descriptions under each step to briefly explain the purpose.
  
This flowchart will provide a high-level overview of your project’s methodology, capturing all the essential steps while keeping it organized and easy to understand.


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

