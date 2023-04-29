from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd
from get_tweets import get_all_tweets

model = load("BERT_CNN.joblib")

def requestResults(name):
    tweets = get_all_tweets(name)
    tweets['prediction'] = model.predict(tweets['tweet_text'])
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 50)
    pd.options.display.max_colwidth = 20
    return str(tweets)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        search = request.form['search']
        return redirect(url_for('success', name=search))


@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "


if __name__ == '__main__' :
    app.run(debug=True)