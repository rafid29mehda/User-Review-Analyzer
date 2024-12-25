### Script for "Dataset Used" Slide

"Let’s begin by discussing the datasets used in our study. First, we have the **PROMISE dataset**, a well-known resource in the software requirements domain. This dataset contains a total of **969 requirements**, split into **444 functional requirements (FRs)** and **525 non-functional requirements (NFRs)**. It serves as a balanced and reliable foundation for training the model on labeled data.

To expand the training data and introduce more diversity and scalability, we used an **unlabeled reviews dataset**. This dataset contains over **12,000 reviews** of various app store applications written by real users. The reviews provide a rich source of software requirements that are diverse and dynamic. Using this dataset, we created additional training data by labeling it with the help of the fine-tuned PROMISE model.

In the next slide, we will delve into the data labeling procedure and the preprocessing steps involved in preparing the dataset for training."

---

### Script for "Data Preprocessing" Slide

"Now, let’s move to the first stage of our methodology: **Data Preprocessing**. This stage focused on preparing the unlabeled review dataset by labeling it and ensuring it was ready for training. 

To achieve this, we first fine-tuned the PROMISE dataset using the **BERT-base-uncased model**, which was trained to differentiate between functional and non-functional requirements. We ensured that the dataset was ready for training by adjusting key hyperparameters, such as learning rate, batch size, and the number of epochs. The fine-tuning process enabled us to create a robust classifier capable of labeling the previously unlabeled review dataset.

In this fine-tuned model, we provided the **unlabeled review dataset as input**, the model automatically labeled each review as either functional or non-functionalthe and generated the labeled review dataset as output. This step significantly expanded our training data, ensuring it was both diverse and representative of real-world software requirements.

 In the next slide, we will look at how the labeled review dataset was used to train our proposed model."

---

This approach is now sequential, clear, and aligned with how you’ll speak during the presentation. 
