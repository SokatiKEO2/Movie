import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon') #to find the meanings of words, synonyms, antonyms, and more.

sid = SentimentIntensityAnalyzer() # NLTK :: Natural Language Toolkit function
df = pd.read_csv('./IMDB_Dataset/Test.csv')
df_subset = df.head(500)
def assign_sentiment(row):
    score = sid.polarity_scores(row)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_subset['Sentiment'] = df_subset['text'].apply(assign_sentiment)
#The function assign_sentiment is applied to each row in the 'text' column of a DataFrame df_subset using the apply method.
sentiment_counts = df_subset['Sentiment'].value_counts()
plt.pie(sentiment_counts, labels = sentiment_counts.index, startangle=90, autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sentiment Analysis Results')
plt.show()