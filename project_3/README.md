# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

### Problem Statement

Reddit is a forums where registered members can share news or comments on other member’s posts. Posts are organised by subject into "subreddits" which cover a variety of topics such as news, sports, fitness, politics, religion, video games, music, books, science, movies, cooking, pets, and image-sharing. As data scientists with natural language processing (NLP) knowledge, we are interested to find out if we can use NLP to classify posts from two different subreddits based on their title and text content. Such classification model can be useful for every company to classify categories, sentimental analysis or decisions. However, for this project i will be presenting how NLP can differential two different subreddit accurately to non-technical audience such as subreddit moderators so that they are able to move the contents to the correct subreddit.

### Executive Summary
For this project,  I'll be classifying posts from two subreddits that are somewhat similar:

r/Science: A subreddit channel where Reddit users share and discuss content related to Science topic (Technology, health, engineering, Economics and etc)
r/wallstreetbets: A subreddit channel where Reddit users share and discuss content related to US stocks
Using NLP to analyse the posts from the two subreddits may help to shed some insight or connection between two different topics.

Starting with data extraction, i've collected posts by connecting to PushshiftAPI. This Web API allows us to access the Reddit comment and submission database to extract their data. i have collected posts between 06 Mar 2020 to 06 Mar 2022 because that was the timeline whereby Covid-19 started to surface. I've split into 2 extraction for each subreddit to overcome connection broke error. There are 16,000 posts from r/wallstreetbets and 14,000 posts from r/science. The posts collected were then cleaned, pre-processed for exploratory data analysis (EDA) and classification modelling. Based on the insights from our EDA, we zoomed in on the characteristics (i.e the words usage/type of posts) of each subreddit and were able to link the connection between two different topics.

Following the EDA process, we tested and evaluated three classification models - Logistic Regression, Naive Bayes & Random Forest Classifier with two different NLP vectorizers - Count Vectorizer & TF-IDF Vectorizer to select the best parameters which produce the best model performance. The Naive Bayes model using TF-IDF Vectorizer was then selected as our production model. Our production model is able to classify posts from the two subreddits with an F1-Score score of 96.7%. We consider our production model to be performing relatively well given that it is better than the baseline model. This simple classifier will be beneficial for content create as an individual or even the subreddit moderators to keep the contents of each subreddit relevant to the community.


### Conclusion and Recommendations

We would recommend the subreddit moderators of r/science and r/wallstreetbets to use our model as it can correctly classify 97.87% of wallstreetbets posts and 98.94% of science posts. The model hence eliminates the need for manual screening currently performed by the subreddit moderators and will free up their time to focus their expertise on more productive tasks. The subreddit moderators can also use the insights from the model to understand the overall direction of their subreddit.

Alternatively, this model can be deployed to be used by the subreddit community too. The model can be integrated into the Reddit platform/ enabled as a browser extension, to help suggest the correct subreddit for the users to publish their post.