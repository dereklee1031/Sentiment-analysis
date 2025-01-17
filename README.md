# Sentiment Analysis with Naive Bayes Classifier

This project implements a Naive Bayes classifier to predict the sentiment (positive or negative) of tweets. The project covers data preprocessing, frequency analysis, model training, and prediction accuracy evaluation.

## Features
- **Text Preprocessing**: Clean tweets by removing irrelevant characters, stemming words, and filtering stop words.
- **Frequency Analysis**: Build frequency dictionaries for word-label pairs.
- **Naive Bayes Classifier**: Train a classifier using frequency dictionaries and evaluate its accuracy on test data.
- **Prediction**: Predict sentiment scores for new tweets and calculate log-likelihood values.

## Installation
1. Clone this repository.
2. Install Python dependencies:
   ```bash
   pip install numpy nltk
   ```
3. Download the NLTK stopwords package:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage
1. **Preprocess Tweets**:
   Use the `process_tweet` function to clean raw tweets by removing unnecessary symbols and stop words.

2. **Train Model**:
   Train the Naive Bayes classifier using the `train_naive_bayes` function with training data.

3. **Predict Sentiment**:
   Use the `naive_bayes_predict` function to predict the sentiment of new tweets:
   ```python
   my_tweet = "This is an amazing project!"
   score = naive_bayes_predict(my_tweet, logprior, loglikelihood)
   print(f"Sentiment Score: {score}")
   ```

4. **Evaluate Accuracy**:
   Evaluate the classifier’s performance using the `test_naive_bayes` function:
   ```python
   accuracy = test_naive_bayes(test_x, test_y, logprior, loglikelihood)
   print(f"Model Accuracy: {accuracy}")
   ```

## Notebook Overview
- **Preprocessing**: Functions for cleaning and tokenizing tweets.
- **Frequency Analysis**: Builds a dictionary of word frequencies by sentiment.
- **Model Training**: Trains a Naive Bayes model to compute log priors and log likelihoods.
- **Prediction and Evaluation**: Predicts sentiments and evaluates model accuracy.

## Example Output
### Sentiment Prediction
```
Input: "I am happy"
Output: Positive Sentiment Score: 2.14
```

### Accuracy
```
Naive Bayes accuracy = 0.9955
```

## Acknowledgments
- This implementation is based on a sentiment analysis framework with Naive Bayes.
