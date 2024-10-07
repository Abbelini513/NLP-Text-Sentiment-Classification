# NLP Assignment #1: Sentiment Analysis and Text Classification

## Overview

This repository contains a project on sentiment analysis and text classification using various approaches to text processing. The repository includes the following files:

- `Bondareva_HW1_NLP.ipynb`: A Jupyter Notebook with full data analysis (EDA), heuristic development, text vectorization, machine learning model training, and evaluation.
- `ml-bio-2024`: The dataset used for this project.

> Note: The fine-tuned BERT model (`fine_tuned_bert`) could not be uploaded to this repository as it exceeds the file size limit.

## Task

The task required the following steps:
1. **Exploratory Data Analysis (EDA)**.
2. **Create a heuristic for predicting the target class**: Using naive methods without applying machine learning.
3. **Implement a text vectorization method and train a model**: Selecting and training one of the classical machine learning models.
4. **Evaluate the model's performance**: Splitting the data into training and validation sets and calculating evaluation metrics.

## 1. Exploratory Data Analysis (EDA)

During the EDA, the following aspects of the data were analyzed:
- **Dataset structure**: Using `Pandas` and `Matplotlib` to review the first rows of data, assess class distributions, and evaluate text lengths.
- **Word frequency visualization**: Word clouds were generated to visually represent the most frequent terms.
- **Data cleaning**: Regular expressions were used to remove unwanted symbols, stop words, and emojis.
- **Lemmatization and normalization**: The `nltk` library was applied for text processing (lemmatization and stop-word removal).

## 2. Heuristic

A simple heuristic model was developed to predict the sentiment of the text based on pre-processed data:
- **Classification by words**: For each text, the number of positive, negative, and neutral words was counted.
- **Heuristic rules**:
  - If the text contains extremely positive words, it is classified as "Extremely Positive".
  - If extremely negative words are present, it is classified as "Extremely Negative".
  - If positive words outnumber negative ones, it is classified as "Positive", and vice versa.
  - If the counts are equal, the text is classified as "Neutral".

## 3. Model Training

The following steps were taken to train machine learning models:
- **Text vectorization methods**: 
  - **Bag of Words (BoW)** and **TF-IDF** were used to transform the text into numerical vectors.
- **Machine learning models**:
  - Multinomial Naive Bayes.
  - Random Forest for text classification.
- **Using a pre-trained model from HuggingFace**:
  - The [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model was fine-tuned on our dataset (ml-bio-2024) for the task of classifying text into 5 sentiment categories.

## 4. Metrics

Model performance was evaluated using the following metrics:
- **Precision**,
- **Recall**,
- **F1-Score**.

Detailed classification reports were generated for each model, allowing for a thorough evaluation of their performance. The best-performing model was the fine-tuned BERT model, which provided the highest Precision, Recall, and F1-Score across all classes.

## Conclusion

This project demonstrated that a custom heuristic model developed in Step 2 achieved the highest accuracy of 60.07% in text classification. Despite its simplicity, it outperformed even classical machine learning methods. However, after introducing additional preprocessing steps, such as negation handling and stop-word removal, the model's accuracy dropped to 57.55%, highlighting that excessive preprocessing can degrade results.

The Bag of Words method achieved 47.20% accuracy but struggled with subtle sentiment distinctions.

The TF-IDF method showed a lower accuracy of 36.63% compared to Bag of Words. Despite the method's strength in weighing words based on their importance and frequency in documents, it did not lead to better model performance. This may indicate that TF-IDF is less effective for sentiment classification, where word context is more important than frequency.

The Naive Bayes classifier, used in a pipeline with TF-IDF vectorization, produced identical results (36.63%) to those of the TF-IDF approach alone. This is explained by the fact that the data structure and representation method remained unchanged.

In Step 3, Word2Vec text vectorization was used, followed by the training of Random Forest and SVC models. Both models achieved around 40% accuracy, with low F1-scores, indicating that these models were not able to reliably predict sentiment across most classes in our task.

As a final step, a pre-trained BERT model (Twitter-roBERTa-base for Sentiment Analysis) from HuggingFace was fine-tuned on our dataset.

The BERT model achieved the highest accuracy — 88.66%, with strong Precision, Recall, and F1-Score values (around 0.90 for all classes). The model reliably recognized texts from all categories, including complex emotional classes such as "Extremely Positive" and "Extremely Negative". This made it the best-performing model in this assignment.

## Libraries Used

The following libraries were used in this project:
- `Pandas`, `NumPy` for data handling.
- `Matplotlib`, `Seaborn` for data visualization.
- `nltk`, `re`, `WordCloud` for text processing.
- `Scikit-learn` for vectorization and machine learning models.
- `Transformers` for working with the BERT model.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook to run and analyze the code:
    ```bash
    jupyter notebook Bondareva_HW1_NLP.ipynb
