""" 
Requires tweet-preprocessor
WARNING: Hardcoded paths, Works only for the davidson hate speech dataset
"""

import preprocessor as p
import pandas as pd
import re
from tqdm import tqdm

df = pd.read_csv("./twitter.csv")

tweets = []
for i in tqdm(range(len(df))):
    clean_tweet = p.clean(df["tweet"][i])
    clean_tweet.strip()
    clean_tweet = re.sub(r'[^\w]', ' ', clean_tweet)
    clean_tweet = clean_tweet.replace("amp", "")
    clean_tweet = re.sub(' +', ' ', clean_tweet)

    if df["class"][i] == 2:
        label = 0
    else:
        label = 1

    tweets.append((label, clean_tweet))

with open("./cleaned_tweets.tsv", "w") as f:
    for i in range(len(tweets)):
        f.write(str(tweets[i][0]) + "\t" + tweets[i][1] + "\n")

