# 📊 Data Science Portfolio — 12 Live Model Demos

Interactive demos of machine learning models trained across a 30-project
data science portfolio covering regression, classification, clustering,
computer vision, NLP, recommendation systems, time series, data visualization,
deep learning, and AI agents.

Built with [Streamlit](https://streamlit.io) and deployed on
[Streamlit Community Cloud](https://share.streamlit.io).

**Demos in this app:**

| Demo | Model | Topic |
|---|---|---|
| Insurance Cost | Gradient Boosting | Regression |
| Fuel Efficiency (MPG) | Random Forest | Regression |
| Heart Disease Risk | Logistic Regression | Classification |
| Wine Quality | Random Forest | Classification |
| Customer Churn | Balanced Logistic Regression | Classification |
| Customer Segments | K-Means | Clustering |
| SMS Spam Filter | Naive Bayes + Bag of Words | NLP |
| Review Sentiment | TF-IDF bigrams + LogReg | NLP |
| Language Detection | Char n-grams + Naive Bayes (22 languages) | NLP |
| Movie Recommender | Genre TF-IDF + cosine similarity | Recommenders |
| Book Recommender | Weighted rating + content similarity | Recommenders |
| Chatbot | Retrieval agent with confidence threshold | AI Agents |

All models were trained in Jupyter notebooks with full EDA, honest baseline
comparisons, and documented pitfalls (e.g. why stop-word removal breaks
sentiment analysis, why LSTMs can't beat the naive baseline on stock prices).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
