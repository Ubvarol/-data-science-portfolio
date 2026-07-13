"""
Data Science Portfolio — 30 Projects, 12 Live Demos
Interactive Streamlit app serving models trained in the project notebooks.
"""

import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Data Science Portfolio", page_icon="📊", layout="centered")


@st.cache_resource
def load_model(name):
    return joblib.load(f"models/{name}.joblib")


PAGES = [
    "🏠 Home",
    "💰 Insurance Cost (Regression)",
    "⛽ Fuel Efficiency (Regression)",
    "❤️ Heart Disease (Classification)",
    "🍷 Wine Quality (Classification)",
    "📞 Customer Churn (Classification)",
    "🛍️ Customer Segments (Clustering)",
    "📩 SMS Spam Filter (NLP)",
    "😊 Sentiment Analysis (NLP)",
    "🌍 Language Detection (NLP)",
    "🎬 Movie Recommender",
    "📚 Book Recommender",
    "🤖 Chatbot (AI Agent)",
]
page = st.sidebar.radio("Choose a demo", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Every demo runs a model trained in the accompanying Jupyter notebooks "
    "(30 projects across 10 data science topics)."
)

# ----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📊 Data Science Portfolio")
    st.markdown("""
This Space hosts **live demos of 12 machine learning models** trained across a
30-project data science portfolio:

| Topic | Projects |
|---|---|
| Regression | House prices, insurance costs, fuel efficiency |
| Classification | Heart disease, customer churn, wine quality |
| Clustering | Customer segments, credit cards, music |
| Computer Vision | Digit recognition, face detection, pencil sketch |
| NLP | Spam filter, sentiment analysis, language detection |
| Recommenders | Movies (content + collaborative), books |
| Time Series | ARIMA forecasting, stock analysis, daily births |
| Data Visualization | Netflix catalog, bar chart race, Uber trips |
| Deep Learning | CNN image classifier, tabular NN, LSTM |
| AI Agents | Retrieval chatbot, minimax game agent, Q-learning |

👈 **Pick a demo from the sidebar** and try the models with your own inputs.
""")

# ----------------------------------------------------------------------------
elif page.startswith("💰"):
    st.title("💰 Medical Insurance Cost Prediction")
    st.caption("Gradient Boosting pipeline · trained on 1,338 insurance records · R² ≈ 0.88")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 64, 35)
        bmi = st.slider("BMI (body mass index)", 15.0, 50.0, 27.5, 0.1)
        children = st.slider("Number of children", 0, 5, 0)
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    person = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi,
                            "children": children, "smoker": smoker, "region": region}])
    cost = load_model("insurance_cost_gb").predict(person)[0]
    st.metric("Predicted annual insurance cost", f"${cost:,.0f}")
    if smoker == "yes":
        no_smoke = person.assign(smoker="no")
        saved = cost - load_model("insurance_cost_gb").predict(no_smoke)[0]
        st.info(f"Quitting smoking would lower the prediction by **${saved:,.0f}** per year.")

# ----------------------------------------------------------------------------
elif page.startswith("⛽"):
    st.title("⛽ Car Fuel Efficiency (MPG) Prediction")
    st.caption("Random Forest pipeline · trained on the Auto MPG dataset · R² ≈ 0.90")

    col1, col2 = st.columns(2)
    with col1:
        cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.slider("Displacement (cu in)", 60.0, 460.0, 120.0)
        horsepower = st.slider("Horsepower", 45.0, 230.0, 90.0)
    with col2:
        weight = st.slider("Weight (lbs)", 1600, 5200, 2500)
        acceleration = st.slider("0-60 mph time (s)", 8.0, 25.0, 15.0)
        origin = st.selectbox("Origin", ["usa", "europe", "japan"])

    car = pd.DataFrame([{"cylinders": cylinders, "displacement": displacement,
                         "horsepower": horsepower, "weight": weight,
                         "acceleration": acceleration, "model_year": 80, "origin": origin}])
    mpg = load_model("fuel_mpg_rf").predict(car)[0]
    st.metric("Predicted fuel efficiency", f"{mpg:.1f} MPG",
              help="Miles per gallon — higher is more efficient")
    st.progress(min(mpg / 45, 1.0))

# ----------------------------------------------------------------------------
elif page.startswith("❤️"):
    st.title("❤️ Heart Disease Risk Prediction")
    st.caption("Logistic Regression pipeline · UCI Heart dataset · ~85% accuracy")
    st.warning("Educational demo only — never a substitute for medical advice.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 29, 77, 54)
        sex = 1 if st.selectbox("Sex", ["male", "female"]) == "male" else 0
        cp = st.selectbox("Chest pain type", [0, 1, 2, 3],
                          help="0: typical angina · 3: asymptomatic")
        trestbps = st.slider("Resting blood pressure", 94, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)
    with col2:
        fbs = 1 if st.selectbox("Fasting blood sugar > 120", ["no", "yes"]) == "yes" else 0
        restecg = st.selectbox("Resting ECG result", [0, 1, 2])
        thalach = st.slider("Max heart rate achieved", 71, 202, 150)
        exang = 1 if st.selectbox("Exercise-induced angina", ["no", "yes"]) == "yes" else 0
    with col3:
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        slope = st.selectbox("ST slope", [0, 1, 2])
        ca = st.selectbox("Major vessels colored (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    patient = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]],
                           columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                    "restecg", "thalach", "exang", "oldpeak",
                                    "slope", "ca", "thal"])
    prob = load_model("heart_disease_logreg").predict_proba(patient)[0][1]
    st.metric("Heart disease probability", f"{prob:.0%}")
    st.progress(prob)

# ----------------------------------------------------------------------------
elif page.startswith("🍷"):
    st.title("🍷 Wine Quality Classifier")
    st.caption("Random Forest · UCI red wine dataset · 'good' = taster score ≥ 7")

    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity = st.slider("Fixed acidity", 4.6, 15.9, 8.3)
        volatile_acidity = st.slider("Volatile acidity", 0.12, 1.58, 0.5)
        citric_acid = st.slider("Citric acid", 0.0, 1.0, 0.27)
        residual_sugar = st.slider("Residual sugar", 0.9, 15.5, 2.5)
        chlorides = st.slider("Chlorides", 0.01, 0.61, 0.08)
        free_so2 = st.slider("Free sulfur dioxide", 1.0, 72.0, 16.0)
    with col2:
        total_so2 = st.slider("Total sulfur dioxide", 6.0, 289.0, 46.0)
        density = st.slider("Density", 0.990, 1.004, 0.9967, format="%.4f")
        ph = st.slider("pH", 2.74, 4.01, 3.31)
        sulphates = st.slider("Sulphates", 0.33, 2.0, 0.66)
        alcohol = st.slider("Alcohol (%)", 8.4, 14.9, 10.4)

    wine = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_so2, total_so2, density, ph, sulphates, alcohol]],
                        columns=["fixed acidity", "volatile acidity", "citric acid",
                                 "residual sugar", "chlorides", "free sulfur dioxide",
                                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
    prob = load_model("wine_quality_rf").predict_proba(wine)[0][1]
    verdict = "🍷 Good wine!" if prob >= 0.5 else "🫤 Average wine"
    st.metric(verdict, f"{prob:.0%} probability of being good")
    st.caption("Tip: alcohol content is the strongest single predictor — try the slider!")

# ----------------------------------------------------------------------------
elif page.startswith("📞"):
    st.title("📞 Customer Churn Prediction")
    st.caption("Balanced Logistic Regression · Telco dataset (7,043 customers) · recall-oriented")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with col3:
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)",
                                                  "Credit card (automatic)"])
        security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])

    customer = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": support,
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
    }])
    prob = load_model("customer_churn_logreg").predict_proba(customer)[0][1]
    st.metric("Churn probability", f"{prob:.0%}")
    st.progress(prob)
    if prob >= 0.5:
        st.error("High churn risk — retention offer recommended.")
    else:
        st.success("Low churn risk.")

# ----------------------------------------------------------------------------
elif page.startswith("🛍️"):
    st.title("🛍️ Customer Segmentation")
    st.caption("K-Means (k=5) · mall customer data · segments from income and spending")

    income = st.slider("Annual income (k$)", 15, 140, 60)
    spending = st.slider("Spending score (1-100)", 1, 100, 50)

    customer = pd.DataFrame([[income, spending]], columns=["income", "spending_score"])
    cluster = int(load_model("customer_segmentation_kmeans").predict(customer)[0])

    # Segment descriptions derived from the notebook's cluster profiling
    pipe = load_model("customer_segmentation_kmeans")
    centers = pipe.named_steps["scaler"].inverse_transform(
        pipe.named_steps["kmeans"].cluster_centers_)

    def describe(center):
        inc, sp = center
        inc_lvl = "high" if inc > 75 else ("low" if inc < 45 else "average")
        sp_lvl = "high" if sp > 65 else ("low" if sp < 35 else "average")
        names = {
            ("high", "high"): "🌟 VIP / target customers",
            ("high", "low"): "🧐 Careful rich",
            ("low", "high"): "🎢 Enthusiast spenders",
            ("low", "low"): "🪙 Budget-conscious",
        }
        return names.get((inc_lvl, sp_lvl), "🙂 Standard customers")

    st.metric("Assigned segment", f"Cluster {cluster} — {describe(centers[cluster])}")
    profile = pd.DataFrame(centers, columns=["Income (k$)", "Spending score"]).round(1)
    profile["Segment"] = [describe(c) for c in centers]
    profile.index.name = "Cluster"
    st.dataframe(profile)

# ----------------------------------------------------------------------------
elif page.startswith("📩"):
    st.title("📩 SMS Spam Filter")
    st.caption("Naive Bayes + Bag of Words · 5,572 SMS messages · ~99% accuracy")

    text = st.text_area("Paste an SMS message:",
                        "Congratulations! You won a free iPhone. Click here to claim your prize now!")
    if text.strip():
        model = load_model("sms_spam_nb")
        prob = model.predict_proba([text])[0][1]
        if prob >= 0.5:
            st.error(f"🚨 SPAM (probability {prob:.0%})")
        else:
            st.success(f"✅ Legitimate message (spam probability {prob:.0%})")

# ----------------------------------------------------------------------------
elif page.startswith("😊"):
    st.title("😊 Product Review Sentiment")
    st.caption("TF-IDF bigrams + Logistic Regression · Flipkart reviews · ~98% accuracy")

    text = st.text_area("Write a product review:",
                        "The battery life is not good at all, very disappointed.")

    def clean_text(t):
        t = t.lower().replace("read more", " ")
        t = re.sub(r"[^a-z\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    if text.strip():
        prob = load_model("sentiment_logreg").predict_proba([clean_text(text)])[0][1]
        if prob >= 0.5:
            st.success(f"😊 Positive ({prob:.0%})")
        else:
            st.error(f"😠 Negative (positive probability {prob:.0%})")
        st.caption('Fun fact from the notebook: this model keeps the word "not" on purpose — '
                   "standard stop-word removal would destroy negation!")

# ----------------------------------------------------------------------------
elif page.startswith("🌍"):
    st.title("🌍 Language Detection")
    st.caption("Character n-grams + Naive Bayes · 22 languages · ~98% accuracy")

    text = st.text_area("Type a sentence in any of 22 languages:",
                        "Bugün hava çok güzel, yürüyüşe çıkalım!")
    if text.strip():
        model = load_model("language_detection_nb")
        probs = model.predict_proba([text])[0]
        top3 = pd.Series(probs, index=model.classes_).nlargest(3)
        st.metric("Detected language", top3.index[0])
        st.dataframe(top3.rename("probability").round(3))

# ----------------------------------------------------------------------------
elif page.startswith("🎬"):
    st.title("🎬 Movie Recommender (Content-Based)")
    st.caption("Genre TF-IDF + cosine similarity · MovieLens (9,742 movies)")

    art = load_model("movie_recommender_content")
    movies, matrix = art["movies"].reset_index(drop=True), art["genre_matrix"]

    popular = movies[movies["num_ratings"] >= 30].sort_values("title")
    title = st.selectbox("Pick a movie you like:", popular["title"].tolist(),
                         index=popular["title"].tolist().index("Toy Story (1995)")
                         if "Toy Story (1995)" in popular["title"].tolist() else 0)

    idx = movies[movies["title"] == title].index[0]
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    recs = (movies.assign(similarity=sims).drop(idx)
            .query("num_ratings >= 30")
            .sort_values(["similarity", "avg_rating"], ascending=False)
            .head(10))
    st.write(f"**Because you liked** *{title}*:")
    st.dataframe(recs[["title", "genres", "avg_rating"]].round(2), hide_index=True)

# ----------------------------------------------------------------------------
elif page.startswith("📚"):
    st.title("📚 Book Recommender")
    st.caption("Author/title similarity + IMDb-style weighted rating · goodbooks-10k")

    art = load_model("book_recommender")
    books, matrix = art["books"].reset_index(drop=True), art["content_matrix"]

    choice = st.selectbox("Pick a book you like:",
                          books.nlargest(2000, "ratings_count").sort_values("title")["title"].tolist())
    idx = books[books["title"] == choice].index[0]
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    recs = (books.assign(similarity=sims).drop(idx)
            .query("similarity > 0.1")
            .sort_values(["similarity", "weighted_rating"], ascending=False)
            .head(10))
    st.write(f"**Because you liked** *{choice}*:")
    st.dataframe(recs[["title", "authors", "average_rating"]].round(2), hide_index=True)

# ----------------------------------------------------------------------------
elif page.startswith("🤖"):
    st.title("🤖 Data Science Chatbot")
    st.caption("Retrieval agent: TF-IDF + cosine similarity with a confidence threshold")

    bot = load_model("chatbot_retrieval")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "Hi! Ask me about ML concepts — overfitting, clustering, TF-IDF, precision vs recall...")]

    for role, msg in st.session_state.chat_history:
        st.chat_message("assistant" if role == "bot" else "user").write(msg)

    if question := st.chat_input("Ask a data science question"):
        st.session_state.chat_history.append(("user", question))
        vec = bot["vectorizer"].transform([question.lower()])
        sims = cosine_similarity(vec, bot["question_matrix"]).ravel()
        best = sims.argmax()
        if sims[best] < 0.3:
            answer = ("Sorry, I don't know that one yet. Try asking about ML concepts "
                      "like overfitting, clustering, or neural networks.")
        else:
            answer = bot["answers"][best]
        st.session_state.chat_history.append(("bot", answer))
        st.rerun()
