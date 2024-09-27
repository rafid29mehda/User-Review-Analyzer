**Methodology**

This study presents a hybrid model designed to enhance the classification accuracy of user reviews by integrating embeddings from a fine-tuned Doc2Vec model and a fine-tuned DistilBERT model. The methodology involves several key stages: data collection and preprocessing, handling class imbalance, fine-tuning of models with hyperparameter optimization, generation of embeddings using alternative pooling strategies, feature combination and dimensionality reduction, ensemble modeling, and comprehensive evaluation.

---

**Data Collection and Preprocessing**

The dataset utilized comprises user reviews from a specific application, each annotated as either *Functional* (F) or *Non-Functional* (NF). The data were imported into a structured format using a Pandas DataFrame, facilitating systematic preprocessing and analysis.

To prepare the data for modeling, categorical labels were transformed into binary numerical values to suit supervised learning algorithms. Specifically, reviews labeled as *Functional* were assigned a value of 1, while those labeled as *Non-Functional* were assigned a value of 0:

\[
\text{Label} =
  \begin{cases}
    1, & \text{if Functional (F)} \\
    0, & \text{if Non-Functional (NF)}
  \end{cases}
\]

Textual preprocessing was conducted to standardize the reviews and enhance the quality of the input data. The preprocessing steps included converting all text to lowercase to ensure uniformity, removing punctuation and special characters to reduce noise, tokenizing the text into individual words or tokens, and lemmatizing words to reduce them to their base forms. This preprocessing pipeline aimed to normalize the textual data, reduce dimensionality, and improve the subsequent embedding representations.

---

**Handling Class Imbalance**

An initial examination of the dataset revealed an imbalance between the *Functional* and *Non-Functional* classes, which could potentially bias the model towards the majority class. To address this issue, both resampling techniques and class weight adjustments were employed.

Resampling techniques involved using the Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic examples for the minority class, thereby balancing the class distribution. Additionally, random undersampling was applied to the majority class to reduce its size and match the number of instances in the minority class. These methods aimed to create a more balanced dataset without significant loss of information.

Class weight adjustments were also implemented within the XGBoost classifier by modifying the `scale_pos_weight` parameter. This parameter was set based on the inverse ratio of class frequencies:

\[
\text{scale\_pos\_weight} = \frac{\text{Number of Negative Class Instances}}{\text{Number of Positive Class Instances}}
\]

Adjusting this parameter penalizes the misclassification of the minority class more heavily during training, encouraging the model to pay equal attention to both classes.

---

**Fine-Tuning and Hyperparameter Optimization**

The Doc2Vec model was fine-tuned using the Gensim library, employing the Distributed Bag of Words (DBOW) architecture known for its effectiveness in capturing document-level semantics. Hyperparameter tuning was performed through grid search over key parameters, such as vector size (ranging from 100 to 300), window size (ranging from 5 to 15 words), and minimum word count (ranging from 1 to 5). The optimal hyperparameters were selected based on performance metrics evaluated on a validation set, ensuring that the model effectively captured the semantic representations of the reviews.

Simultaneously, the DistilBERT model was fine-tuned using the Hugging Face Transformers library to capture contextual word-level information. Alternative pooling strategies were explored to obtain fixed-size sentence embeddings. The three pooling methods considered were:

1. **[CLS] Token Embedding**: Extracting the embedding corresponding to the `[CLS]` token, which is designed to capture the aggregate representation of the sequence.

2. **Mean Pooling**: Calculating the average of all token embeddings across the sequence, providing a generalized representation of the text:

   \[
   \text{Embedding}_{\text{mean}} = \frac{1}{n} \sum_{i=1}^{n} h_i
   \]

   where \( h_i \) represents the hidden state of the \( i \)-th token, and \( n \) is the total number of tokens.

3. **Max Pooling**: Selecting the maximum value across all token embeddings, capturing the most significant features:

   \[
   \text{Embedding}_{\text{max}} = \max_{i \in \{1, \dots, n\}} h_i
   \]

Hyperparameter optimization for the DistilBERT model involved tuning parameters such as learning rate (ranging from \( 2 \times 10^{-5} \) to \( 5 \times 10^{-5} \)), batch size (16, 32, 64), and the number of epochs (3 to 10). Grid search and cross-validation were employed to systematically explore the hyperparameter space. Early stopping was implemented by monitoring the validation loss and halting training when no improvement was observed, thus preventing overfitting.

---

**Embedding Generation and Feature Combination**

Following the fine-tuning of both models, embeddings were generated for each review. The Doc2Vec model produced vector representations that encapsulated the semantic content at the document level. The DistilBERT model, utilizing the alternative pooling strategies, generated contextual embeddings that captured nuanced word-level information.

To leverage the strengths of both models, the embeddings were concatenated to form a comprehensive feature vector for each review:

\[
\text{Hybrid Embedding} = [ \text{Doc2Vec Embedding}; \text{Embedding}_{\text{CLS}}; \text{Embedding}_{\text{mean}}; \text{Embedding}_{\text{max}} ]
\]

Given the high dimensionality of the combined embeddings, Principal Component Analysis (PCA) was applied for dimensionality reduction. PCA aimed to reduce the number of features while retaining 95% of the variance, thus managing computational complexity and mitigating the risk of overfitting.

---

**Ensemble Modeling**

To enhance predictive performance and model robustness, ensemble modeling techniques were employed. Multiple classifiers were trained on the hybrid embeddings, including XGBoost, Support Vector Machine (SVM), Random Forest, and Logistic Regression. Each classifier was selected for its unique strengths in handling different aspects of the data.

An ensemble model was constructed using soft voting, where the predicted probabilities from each individual classifier were averaged to produce the final prediction:

\[
\text{P}_{\text{ensemble}}(y = c) = \frac{1}{M} \sum_{m=1}^{M} \text{P}_{m}(y = c)
\]

where \( M \) is the number of classifiers, and \( \text{P}_{m}(y = c) \) is the predicted probability of class \( c \) by classifier \( m \).

Additionally, stacking was implemented by training a meta-classifier on the outputs of the base classifiers. This approach allowed the meta-classifier to learn how to best combine the predictions of the base models, potentially capturing complex patterns that individual classifiers might miss.

Hyperparameter tuning was performed for each classifier using grid search with cross-validation. This process ensured that the optimal settings for each model were identified, contributing to the overall performance of the ensemble.

---

**Classifier Training and Evaluation**

The dataset was partitioned into training, validation, and test sets in a stratified manner to maintain class distribution, with 70% allocated for training, 15% for validation, and 15% for testing. Each classifier, including the ensemble models, was trained on the training set using the optimized hyperparameters. Early stopping and regularization techniques were applied where appropriate to prevent overfitting.

The performance of the models was evaluated using multiple metrics to provide a comprehensive assessment. Accuracy measured the overall correctness of the predictions, while precision, recall, and F1-score provided insights into the models' performance on each class. The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) was also calculated to evaluate the classifiers' ability to distinguish between the two classes.

Predictions were made on both the validation and test sets. The evaluation metrics were computed, and models were compared to select the best-performing one based on validation performance. The selected model was then subjected to final evaluation on the test set. A confusion matrix was generated to visualize the performance in terms of true positives, false positives, true negatives, and false negatives, offering further insight into the types of errors made by the model.

---

**Reproducibility and Implementation Details**

The implementation utilized several libraries and tools. Gensim was employed for Doc2Vec implementation, while the Transformers library and PyTorch facilitated the fine-tuning and embedding generation of the DistilBERT model. Scikit-learn provided functions for data preprocessing, model training, hyperparameter tuning, and evaluation. XGBoost was used for the XGBoost classifier, and the Imbalanced-learn library was utilized for resampling techniques such as SMOTE.

Experiments were conducted on a machine equipped with a GPU to accelerate the training of neural network models. Random seeds were set for NumPy, PyTorch, and other libraries to ensure reproducibility of results:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Embeddings were generated in batches to optimize memory usage and computational efficiency, particularly important given the size of the datasets and the complexity of the models involved.

---

**Summary**

In summary, the proposed methodology integrates advanced techniques to enhance the classification of user reviews. By thoroughly preprocessing the data and addressing class imbalance, the quality of the input data was significantly improved. Fine-tuning both the Doc2Vec and DistilBERT models with optimized hyperparameters allowed for the extraction of rich embeddings that capture both global semantics and contextual nuances.

The exploration of alternative pooling strategies enriched the feature representations, and the combination of these embeddings into a hybrid feature vector leveraged the strengths of both models. Dimensionality reduction through PCA managed the high dimensionality of the data, reducing computational complexity without substantial loss of information.

Ensemble modeling, including soft voting and stacking, improved predictive performance by combining the strengths of multiple classifiers. Rigorous evaluation using comprehensive metrics ensured the reliability and validity of the model's performance.

This methodology demonstrates a comprehensive approach to classifying user reviews, achieving higher accuracy and generalization by integrating multiple embedding techniques and advanced modeling strategies. The integration of Doc2Vec and DistilBERT embeddings, along with alternative pooling strategies, provides a robust representation of the textual data, enhancing the model's predictive capabilities.

---

**Flowchart of the Methodology**

A flowchart illustrating the steps of the methodology is presented in Figure 1.

![Flowchart of the Proposed Methodology](https://example.com/flowchart.png)

**Figure 1:** Flowchart of the proposed methodology, outlining the sequential steps from data collection to model evaluation.

---

**Considerations for Future Work**

While the enhanced methodology significantly improves classification performance, there are opportunities for future research to further refine and extend this work. Investigating the use of deep learning classifiers, such as feedforward or recurrent neural networks applied to the hybrid embeddings, could capture nonlinear relationships and potentially improve performance.

Applying feature selection techniques, like recursive feature elimination, may identify the most influential features, improving model interpretability and possibly enhancing predictive accuracy. Testing the model's generalization capabilities by adapting it to classify reviews from different domains or languages could assess its robustness and applicability in broader contexts.

Finally, integrating the model into live systems for real-time classification and feedback would be a valuable step toward practical implementation, providing insights into its performance in operational environments and informing further refinements.

---

**Conclusion**

The methodology presented in this study offers a robust framework for enhancing the classification accuracy of user reviews. By combining embeddings from fine-tuned Doc2Vec and DistilBERT models, employing alternative pooling strategies, and utilizing ensemble modeling techniques, the approach effectively captures both the semantic and contextual intricacies of textual data. Comprehensive preprocessing, careful handling of class imbalance, and rigorous hyperparameter optimization contribute to the model's superior performance.

This work demonstrates the potential of integrating multiple natural language processing techniques to address complex classification tasks. The findings suggest that such hybrid models can significantly improve predictive capabilities, offering valuable insights for future research and practical applications in the field of sentiment analysis and beyond.
