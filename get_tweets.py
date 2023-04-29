import tweepy
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

# api key
api_key = "Enter API Key"
# api secret key
api_secret_key = "Enter API Secret Key"
# access token
access_token = "Enter Access Token"
# access token secret
access_token_secret = "Enter Access Token Secret"

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)

def get_all_tweets(text_query):
    tweets_list = []
    count = 20
    for tweet in api.search_tweets(q=text_query, count=count):
        print(tweet.text)
        tweets_list.append({'created_at': tweet.created_at,
                            'tweet_id': tweet.id,
                            'tweet_text': tweet.text})
    return pd.DataFrame.from_dict(tweets_list)
