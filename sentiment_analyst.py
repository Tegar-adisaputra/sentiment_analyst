from flask import Flask, jsonify, request
import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob as tb
from googletrans import Translator
import numpy as np

app = Flask(__name__)
translated = Translator()

@app.route('/')
def index():
    return "Welcome!"

@app.route('/sentiment', methods=['POST'])
def analyst():
    query = request.form.get('keyword')
    translated_keyword = translated.translate(query, dest='en').text
    data = {'keyword': query, 'sentiment': None, 'polarity': None}
    try:
        result = tb(translated_keyword)
        if result.sentiment.polarity > 0.0:
            data['sentiment'] = "positive"
        elif result.sentiment.polarity == 0.0:
            data['sentiment'] = "neutral"
        else:
            data['sentiment'] = "negative"
        data['polarity'] = result.sentiment.polarity
    except:
        return jsonify({"error": "Sorry, there is an error"})
    return jsonify(data)



@app.route('/twitter_scrap', methods=['POST'])
def sentiment_analysis():
    query = request.form.get('keyword')
    tweets = []
    limit = int(request.form.get('limit'))

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.username, tweet.content])
        
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    sentiments = []

    for index, row in df.iterrows():
        tweet_prop = {}
        tweet_prop["Date"] = row['Date']
        tweet_prop["User"] = row['User']
        tweet_prop["Tweet"] = row['Tweet']
        clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(row['Tweet'])).split())
        analyst = tb(clean_tweet)
        if analyst.sentiment.polarity > 0.0:
            tweet_prop["Sentiment"] = "positive"
        elif analyst.sentiment.polarity == 0.0:
            tweet_prop["Sentiment"] = "neutral"
        else:
            tweet_prop["Sentiment"] = "negative"
        sentiments.append(tweet_prop)

    result = pd.DataFrame(sentiments)
    data = result.to_json(orient='index')
    positive = int((result["Sentiment"] == "positive").sum())
    negative = int((result["Sentiment"] == "negative").sum())
    neutral = int((result["Sentiment"] == "neutral").sum())

    labels = ['positive', 'negative', 'neutral']
    counts = [positive, negative, neutral]

    sentiment_distribution = dict(zip(labels, counts))

    return jsonify(sentiment_distribution=sentiment_distribution, data=data)


if __name__ == '__main__':
    app.run(debug=True)
