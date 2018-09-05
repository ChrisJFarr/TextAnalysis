from src.text_processing import normalize_corpus, tokenize_text
import pandas as pd
import numpy as np
import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
import time

""" Load and inspect data """
# https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
df = pd.read_csv("womens_clothing_ecommerce_reviews.csv", index_col=0)
df.columns = [col.replace(" ", "_").lower() for col in df.columns]  # Standardize column names
print(df.isna().sum())

# Keep only rows with complete review field
keep_index = ~df.review_text.isna()
df = df.loc[keep_index, :]

# Using only review_text, how accurately can we predict the rating?
data = df.loc[:, ["review_text", "rating"]]

# Hold out a 10% validation set for performance benchmarking
train_data, validation_data = train_test_split(data, test_size=.10, random_state=0)
train_data, validation_data = train_data.copy(), validation_data.copy()

# Preprocess text
train_text_norm = normalize_corpus(train_data.review_text)
# Extract target
train_target = train_data.rating
# Three examples of ways to process bag of words

# CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1, 1),
                                   stop_words="english",
                                   max_df=.95,
                                   min_df=.05)
train_count_vectorized = count_vectorizer.fit_transform(train_text_norm)

# Cross-validate Predict
model = RandomForestRegressor(n_estimators=10, max_depth=5)
predict = cross_val_predict(model, train_count_vectorized, train_target)
predict_processed = [int(round(i)) for i in predict]

# Analyze accuracy
print("Accuracy: %.2f percent" % (accuracy_score(train_target, predict_processed) * 100))
# Analyze Root Mean Squared Error
print("RMSE: %.2f " % np.sqrt(mean_squared_error(train_target, predict)))

# TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                   stop_words="english",
                                   max_df=.95,
                                   min_df=.05)
train_tfidf_vectorized = tfidf_vectorizer.fit_transform(train_text_norm)

# Cross-validate Predict
model = RandomForestRegressor(n_estimators=10, max_depth=5)
predict = cross_val_predict(model, train_tfidf_vectorized, train_target)
predict_processed = [int(round(i)) for i in predict]

# Analyze accuracy
print("Accuracy: %.2f percent" % (accuracy_score(train_target, predict_processed) * 100))
# Analyze Root Mean Squared Error
print("RMSE: %.2f " % np.sqrt(mean_squared_error(train_target, predict)))


# Tuning parameters (pipeline?)

pipeline = Pipeline([("vect", TfidfVectorizer(max_features=7000)),
                     ("model", RandomForestRegressor(random_state=0))])

parameters = {
    "vect__ngram_range": [(1, 2)],
    "vect__max_df": [.95],
    "vect__min_df": [.05],
    "model__n_estimators": [60, 100],
    "model__max_depth": [30]
}

grid = GridSearchCV(pipeline, parameters, cv=5, n_jobs=3)

start = time.time()
grid.fit(train_text_norm, train_target)
stop = time.time()
print("total time: ", stop-start)

# Analyze results

pprint.pprint(grid.best_params_)

# Generate predictions
model = grid.best_estimator_
predict = cross_val_predict(model, train_text_norm, train_target, cv=5, n_jobs=3)
predict_processed = [int(round(i)) for i in predict]

# Analyze accuracy
print("Accuracy: %.2f percent" % (accuracy_score(train_target, predict_processed) * 100))
# Analyze Root Mean Squared Error
print("RMSE: %.2f " % np.sqrt(mean_squared_error(train_target, predict)))


