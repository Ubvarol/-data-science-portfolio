# 📊 Data Science Portfolio — 27 Live Demos

Interactive demos from a 30-project data science portfolio covering
regression, classification, clustering, computer vision, NLP, recommendation
systems, time series, data visualization, deep learning, and AI agents.

Built with [Streamlit](https://streamlit.io) and deployed on
[Streamlit Community Cloud](https://share.streamlit.io).

**Demos in this app:**

| Demo | Model / Technique | Topic |
|---|---|---|
| House Prices | HistGradientBoosting + feature engineering | Regression |
| Insurance Cost | Gradient Boosting | Regression |
| Fuel Efficiency (MPG) | Random Forest | Regression |
| Heart Disease Risk | Logistic Regression | Classification |
| Wine Quality | Random Forest | Classification |
| Customer Churn | Balanced Logistic Regression | Classification |
| Customer Segments | K-Means (k=5) | Clustering |
| Credit Card Segments | K-Means (k=4) on 17 features | Clustering |
| Music Mood Clusters | K-Means (k=4) on Spotify audio features | Clustering |
| Digit Recognition | Support Vector Machine | Computer Vision |
| Face Detection | OpenCV Haar cascade | Computer Vision |
| Pencil Sketch | OpenCV image arithmetic | Computer Vision |
| SMS Spam Filter | Naive Bayes + Bag of Words | NLP |
| Review Sentiment | TF-IDF bigrams + LogReg | NLP |
| Language Detection | Char n-grams + Naive Bayes (22 languages) | NLP |
| Movie Recommender (Content) | Genre TF-IDF + cosine similarity | Recommenders |
| Movie Recommender (Collaborative) | Item-item rating similarity | Recommenders |
| Book Recommender | Weighted rating + content similarity | Recommenders |
| Airline Forecast | SARIMA with seasonal holdout test | Time Series |
| Stock Explorer | Moving averages, returns, volatility | Time Series |
| Daily Births Forecast | ARIMA(2,1,2) fitted live | Time Series |
| Netflix Catalog | Interactive EDA | Data Viz |
| Bar Chart Race | matplotlib animation | Data Viz |
| Uber Trips Map | Geospatial sample + hourly patterns | Data Viz |
| Chatbot | Retrieval agent with confidence threshold | AI Agents |
| Tic-Tac-Toe | Playable minimax (perfect play) | AI Agents |
| Q-Learning Gridworld | Trained Q-table replay | AI Agents |

The remaining 3 projects (Fashion-MNIST CNN, churn neural network, stock LSTM)
are TensorFlow models — kept out of this app to keep it fast on free hosting;
see the notebooks for their full results.

All models were trained in Jupyter notebooks with full EDA, honest baseline
comparisons, and documented pitfalls (e.g. why stop-word removal breaks
sentiment analysis, why LSTMs can't beat the naive baseline on stock prices).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
