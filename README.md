# Project_7_NHIS

## NLP Project for Disaster Tweet Classification

This project uses Natural Language Processing (NLP) and Machine Learning to classify tweets as Disaster or Non-Disaster.

The main purpose of this project is to identify whether a tweet is related to a real disaster or whether disaster-related words are being used in a different context.

## Dataset Information

- Total Tweets: 7,613
- Non-Disaster Tweets: 4,342
- Disaster Tweets: 3,271
- Target:
  - 0 = Non-Disaster
  - 1 = Disaster

## Text Preprocessing

The following preprocessing steps were applied:

- Converted text to lowercase
- Removed URLs and user mentions
- Removed punctuation and special characters
- Removed hashtag (#) symbol while keeping useful text
- Tokenized the cleaned text
- Removed English stopwords
- Removed extra spaces and unnecessary text patterns
- Joined processed tokens back into cleaned text

## Feature Engineering

Different NLP techniques were used to convert text into useful numerical features:

- Bag of Words (BOW)
- TF-IDF Vectorization
- Maximum 5,000 TF-IDF features
- VADER Sentiment Analysis
- Positive, Negative, Neutral and Compound sentiment scores

TF-IDF was used as the main text representation for training the Machine Learning models.

## Machine Learning Models

Three classification models were tested:

- Logistic Regression
- Random Forest
- Multinomial Naive Bayes

## Initial Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.73% | 84.85% | 72.78% | 78.35% |
| Random Forest | 79.78% | 79.52% | 71.25% | 75.16% |
| Multinomial Naive Bayes | 81.22% | 84.07% | 69.42% | 76.05% |

Logistic Regression showed the best overall initial performance among the tested models.

## Train-Test Split

- Training Data: 80% (6,088 tweets)
- Testing Data: 20% (1,523 tweets)
- Stratified splitting was used to maintain similar class distribution.

## Web Application

The final Logistic Regression model was integrated into a Flask web application.

The application allows users to:

- Enter a new tweet
- Predict Disaster or Non-Disaster
- View the prediction confidence score

## Deployment

The Flask application was deployed using Render Cloud.

Live Application:

https://project-7-nhis.onrender.com

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Flask
- Joblib
- GitHub
- Render

## Project Structure

Project_7_NHIS/

- NLP Project for Disaster Tweet Classification(3).ipynb
- app.py
- disaster_tweet_model.pkl
- tfidf_vectorizer.pkl
- requirements.txt
- .python-version
- README.md
- templates/
  - index.html

## Conclusion

NLP and Machine Learning were successfully used to classify tweets into Disaster and Non-Disaster categories.

TF-IDF provided an effective representation of the cleaned tweet text, and Logistic Regression showed strong overall classification performance.

The final model was integrated into a Flask web application and successfully deployed on Render for real-time tweet classification.
