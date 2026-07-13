"""
Data Science Portfolio — 30 Projects, 27 Live Demos
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


@st.cache_data
def load_csv(name, **kwargs):
    return pd.read_csv(f"data/{name}", **kwargs)


PAGES = [
    "🏠 Home",
    "🏡 House Prices (Regression)",
    "💰 Insurance Cost (Regression)",
    "⛽ Fuel Efficiency (Regression)",
    "❤️ Heart Disease (Classification)",
    "🍷 Wine Quality (Classification)",
    "📞 Customer Churn (Classification)",
    "🛍️ Customer Segments (Clustering)",
    "💳 Credit Card Segments (Clustering)",
    "🎵 Music Mood Clusters (Clustering)",
    "🔢 Digit Recognition (Computer Vision)",
    "👤 Face Detection (Computer Vision)",
    "✏️ Pencil Sketch (Computer Vision)",
    "📩 SMS Spam Filter (NLP)",
    "😊 Sentiment Analysis (NLP)",
    "🌍 Language Detection (NLP)",
    "🎬 Movie Recommender (Content)",
    "🤝 Movie Recommender (Collaborative)",
    "📚 Book Recommender",
    "✈️ Airline Forecast (Time Series)",
    "📈 Stock Explorer (Time Series)",
    "👶 Daily Births Forecast (Time Series)",
    "🎞️ Netflix Catalog (Data Viz)",
    "🏁 Bar Chart Race (Data Viz)",
    "🚕 Uber Trips Map (Data Viz)",
    "🤖 Chatbot (AI Agent)",
    "⭕ Tic-Tac-Toe vs Minimax (AI Agent)",
    "🧭 Q-Learning Gridworld (AI Agent)",
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
This Space hosts **27 live demos** from a 30-project data science portfolio:

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
elif page.startswith("🏡"):
    st.title("🏡 California House Price Prediction")
    st.caption("HistGradientBoosting + feature engineering · 1990 census block groups · R² ≈ 0.85")

    col1, col2 = st.columns(2)
    with col1:
        medinc = st.slider("Median income of the area (×$10k/yr)", 0.5, 15.0, 4.0, 0.1)
        houseage = st.slider("Median house age (years)", 1, 52, 25)
        averooms = st.slider("Average rooms per household", 2.0, 10.0, 5.0, 0.1)
        avebedrms = st.slider("Average bedrooms per household", 0.5, 3.0, 1.0, 0.05)
    with col2:
        population = st.slider("Block population", 100, 10000, 1400, 50)
        aveoccup = st.slider("Average household size", 1.0, 6.0, 2.8, 0.1)
        latitude = st.slider("Latitude", 32.5, 42.0, 34.0, 0.05)
        longitude = st.slider("Longitude", -124.5, -114.0, -118.3, 0.05)

    cities = {"San Francisco": (37.7749, -122.4194), "Los Angeles": (34.0522, -118.2437),
              "San Diego": (32.7157, -117.1611), "Sacramento": (38.5816, -121.4944),
              "Fresno": (36.7378, -119.7871)}
    dist = min(np.sqrt((latitude - lat) ** 2 + (longitude - lon) ** 2)
               for lat, lon in cities.values())

    block = pd.DataFrame([{
        "MedInc": medinc, "HouseAge": houseage, "AveRooms": averooms,
        "AveBedrms": avebedrms, "Population": population, "AveOccup": aveoccup,
        "Latitude": latitude, "Longitude": longitude,
        "bedrooms_per_room": avebedrms / averooms,
        "rooms_per_person": averooms / aveoccup,
        "income_per_room": medinc / averooms,
        "dist_nearest_city": dist,
    }])
    price = load_model("house_price_hgb_fe").predict(block)[0]
    st.metric("Predicted median house value", f"${price * 100000:,.0f}",
              help="1990 dollars — the dataset is a classic 1990 census benchmark")
    st.caption(f"Distance to nearest major city: {dist:.2f}° "
               "(the engineered feature that taught the model geography)")

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
elif page.startswith("💳"):
    st.title("💳 Credit Card Customer Segments")
    st.caption("K-Means (k=4) on 17 behavioral features · 8,950 credit card holders")

    info = load_model("credit_card_profiles")
    medians = info["medians"]

    col1, col2 = st.columns(2)
    with col1:
        balance = st.slider("Card balance ($)", 0, 15000, 1500, 100)
        purchases = st.slider("Purchases last year ($)", 0, 20000, 1000, 100)
        cash_advance = st.slider("Cash advances ($)", 0, 15000, 0, 100)
    with col2:
        credit_limit = st.slider("Credit limit ($)", 500, 20000, 4000, 100)
        payments = st.slider("Payments made ($)", 0, 20000, 1500, 100)
        full_payment = st.slider("Share of months paid in full", 0.0, 1.0, 0.1, 0.05)

    row = medians.copy()
    row["BALANCE"] = balance
    row["PURCHASES"] = purchases
    row["CASH_ADVANCE"] = cash_advance
    row["CREDIT_LIMIT"] = credit_limit
    row["PAYMENTS"] = payments
    row["PRC_FULL_PAYMENT"] = full_payment
    customer = pd.DataFrame([row])[info["features"]]

    cluster = int(load_model("credit_card_kmeans").predict(customer)[0])

    profile = info["profile"]

    def describe(c):
        p = profile.loc[c]
        if p["CASH_ADVANCE"] > profile["CASH_ADVANCE"].mean() * 1.5:
            return "🏧 Cash-advance users (borrowers)"
        if p["PURCHASES"] > profile["PURCHASES"].mean() * 1.5:
            return "🛒 Big spenders"
        if p["PRC_FULL_PAYMENT"] > 0.25:
            return "💎 Pay-in-full responsibles"
        return "😴 Low-activity holders"

    st.metric("Assigned segment", f"Cluster {cluster} — {describe(cluster)}")
    shown = profile.copy()
    shown["Segment"] = [describe(c) for c in shown.index]
    st.dataframe(shown)
    st.caption("Unused sliders are filled with the dataset median — "
               "the model needs all 17 features.")

# ----------------------------------------------------------------------------
elif page.startswith("🎵"):
    st.title("🎵 Music Mood Clusters")
    st.caption("K-Means (k=4) on Spotify audio features · 5,000 songs from 2000-2020")

    info = load_model("music_cluster_info")

    col1, col2, col3 = st.columns(3)
    with col1:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6)
        energy = st.slider("Energy", 0.0, 1.0, 0.65)
    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
        loudness = st.slider("Loudness (dB)", -40.0, 0.0, -7.0)
    with col3:
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.07)
        tempo = st.slider("Tempo (BPM)", 50, 210, 120)
        valence = st.slider("Valence (musical positivity)", 0.0, 1.0, 0.5)

    song = pd.DataFrame([[acousticness, danceability, energy, instrumentalness,
                          liveness, loudness, speechiness, tempo, valence]],
                        columns=info["features"])
    cluster = int(load_model("music_clustering_kmeans").predict(song)[0])

    profile = info["profile"]

    def describe(c):
        p = profile.loc[c]
        if p["acousticness"] >= profile["acousticness"].max() * 0.9:
            return "🪕 Acoustic & mellow"
        if p["speechiness"] >= profile["speechiness"].max() * 0.9:
            return "🎤 Rap & spoken word"
        if p["energy"] >= profile["energy"].max() * 0.9:
            return "⚡ High-energy hits"
        return "🎧 Mainstream pop"

    st.metric("Your song lands in", f"Cluster {cluster} — {describe(cluster)}")
    st.write("**Most popular songs in this cluster:**")
    for name, artist in info["examples"][cluster]:
        st.write(f"- *{name}* — {artist}")
    with st.expander("Cluster audio profiles"):
        st.dataframe(profile)

# ----------------------------------------------------------------------------
elif page.startswith("🔢"):
    st.title("🔢 Handwritten Digit Recognition")
    st.caption("Support Vector Machine · sklearn digits (8×8 pixels) · ~99% accuracy")

    from sklearn.datasets import load_digits

    @st.cache_data
    def get_digits():
        d = load_digits()
        return d.images, d.target

    images, targets = get_digits()

    if "digit_idx" not in st.session_state:
        st.session_state.digit_idx = 42
    if st.button("🎲 Pick a random digit"):
        st.session_state.digit_idx = int(np.random.randint(len(images)))
    idx = st.session_state.digit_idx

    noise = st.slider("Add noise (make it harder)", 0.0, 8.0, 0.0, 0.5)
    img = images[idx] + np.random.default_rng(0).normal(0, noise, (8, 8))
    img = np.clip(img, 0, 16)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image((img / 16 * 255).astype(np.uint8), width=160, caption=f"True label: {targets[idx]}")
    with col2:
        model = load_model("digit_recognition_svm")
        pred = int(model.predict(img.reshape(1, -1))[0])
        scores = model.decision_function(img.reshape(1, -1)).ravel()
        if pred == targets[idx]:
            st.success(f"Model says: **{pred}** ✅")
        else:
            st.error(f"Model says: **{pred}** ❌ (true: {targets[idx]})")
        st.bar_chart(pd.Series(scores, index=range(10), name="decision score"))
    st.caption("The bar chart shows the SVM's confidence for each digit 0-9. "
               "Crank up the noise to find the model's breaking point!")

# ----------------------------------------------------------------------------
elif page.startswith("👤"):
    st.title("👤 Face Detection")
    st.caption("OpenCV Haar cascade — the classic 2001 Viola-Jones detector, no deep learning")

    import cv2

    uploaded = st.file_uploader("Upload a photo (or use the demo below)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread("assets/solvay_1927.jpg")
        st.caption("Demo photo: the 1927 Solvay Conference — Einstein, Curie, Bohr, "
                   "Schrödinger, Heisenberg... how many will the model find?")

    scale = st.slider("Detector sensitivity (scaleFactor)", 1.05, 1.4, 1.1, 0.01)
    neighbors = st.slider("Strictness (minNeighbors)", 1, 10, 5)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)

    out = bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f"Faces found: {len(faces)}")
    st.caption("Lower strictness finds more faces but also more false positives — "
               "the precision/recall trade-off in one slider.")

# ----------------------------------------------------------------------------
elif page.startswith("✏️"):
    st.title("✏️ Pencil Sketch Filter")
    st.caption("Pure OpenCV image math: grayscale → invert → blur → dodge blend")

    import cv2

    uploaded = st.file_uploader("Upload a photo (or use the demo below)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread("assets/building.jpg")

    ksize = st.slider("Pencil softness (blur kernel)", 5, 61, 21, 2)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(255 - gray, (ksize, ksize), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original")
    col2.image(sketch, caption="Pencil sketch")
    st.caption("No machine learning here — just clever pixel arithmetic. "
               "Sometimes the simplest tool wins.")

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
elif page.startswith("🤝"):
    st.title("🤝 Movie Recommender (Collaborative Filtering)")
    st.caption('Item-item similarity from real user ratings — "people who liked X also liked..."')

    art = load_model("movie_recommender_collaborative")
    sim, id_to_title = art["item_similarity"], art["id_to_title"]

    titles = id_to_title.loc[sim.index].sort_values()
    default = "Matrix, The (1999)"
    title = st.selectbox("Pick a movie you like:", titles.tolist(),
                         index=titles.tolist().index(default) if default in titles.tolist() else 0)

    movie_id = titles[titles == title].index[0]
    top = sim[movie_id].drop(movie_id).nlargest(10)
    recs = pd.DataFrame({"title": id_to_title.loc[top.index].values,
                         "rating similarity": top.values.round(3)})
    st.write(f"**Users who liked** *{title}* **also rated highly:**")
    st.dataframe(recs, hide_index=True)
    st.caption("Unlike the content-based recommender, this one knows nothing about "
               "genres — it learns purely from rating patterns of 610 users.")

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
elif page.startswith("✈️"):
    st.title("✈️ Airline Passengers Forecast (SARIMA)")
    st.caption("SARIMA · monthly airline passengers 1949-1958 · forecasts 1959-60")

    res = load_model("airline_sarima")
    actual = load_csv("airline_passengers.csv")
    actual["Month"] = pd.to_datetime(actual["Month"])
    actual = actual.set_index("Month")["Passengers"]

    horizon = st.slider("Forecast horizon (months)", 6, 24, 24)
    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int()

    history = actual[actual.index <= mean.index[0]]
    chart = pd.DataFrame({"History": history,
                          "Forecast": mean,
                          "Actual (holdout)": actual[actual.index.isin(mean.index)],
                          "Lower 95%": ci.iloc[:, 0],
                          "Upper 95%": ci.iloc[:, 1]})
    st.line_chart(chart)
    st.caption("The model was trained only on data up to Dec 1958 — the "
               "'Actual (holdout)' line shows how well it predicted a future "
               "it never saw, including the summer peaks (that's the seasonal part of SARIMA).")

# ----------------------------------------------------------------------------
elif page.startswith("📈"):
    st.title("📈 Stock Price Explorer (AAPL)")
    st.caption("Apple daily closes 2013-2018 · moving averages, returns and volatility")

    df = load_csv("aapl.csv", parse_dates=["date"]).set_index("date")

    col1, col2 = st.columns(2)
    fast = col1.slider("Fast moving average (days)", 5, 60, 20)
    slow = col2.slider("Slow moving average (days)", 30, 250, 50)

    chart = pd.DataFrame({"Close": df["close"],
                          f"MA{fast}": df["close"].rolling(fast).mean(),
                          f"MA{slow}": df["close"].rolling(slow).mean()})
    st.line_chart(chart)

    returns = df["close"].pct_change().dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total return", f"{df['close'].iloc[-1] / df['close'].iloc[0] - 1:+.0%}")
    c2.metric("Best day", f"{returns.max():+.1%}")
    c3.metric("Worst day", f"{returns.min():+.1%}")

    st.write("**Rolling 30-day volatility** (annualized):")
    st.area_chart(returns.rolling(30).std() * np.sqrt(252))
    st.caption("Lesson from the notebooks: prices look predictable, "
               "returns look like noise — that's why the LSTM couldn't beat "
               "the naive baseline in the deep learning project.")

# ----------------------------------------------------------------------------
elif page.startswith("👶"):
    st.title("👶 Daily Births Forecast (ARIMA)")
    st.caption("ARIMA(2,1,2) · daily female births, California 1959 · fitted live on load")

    from statsmodels.tsa.arima.model import ARIMA

    series = load_csv("daily_births.csv", parse_dates=["Date"]).set_index("Date")["Births"].asfreq("D")

    @st.cache_resource
    def fit_births(train_size):
        return ARIMA(series.iloc[:train_size], order=(2, 1, 2)).fit()

    train_size = len(series) - 30
    res = fit_births(train_size)

    horizon = st.slider("Forecast horizon (days)", 7, 30, 30)
    fc = res.get_forecast(steps=horizon)

    chart = pd.DataFrame({"History": series.iloc[max(0, train_size - 90):train_size],
                          "Forecast": fc.predicted_mean,
                          "Actual (holdout)": series.iloc[train_size:train_size + horizon]})
    st.line_chart(chart)
    st.caption("Births have no strong daily seasonality, so the forecast quickly "
               "flattens toward the mean — an honest ARIMA behavior on noisy data. "
               "This tiny model is fitted from raw data every time the app starts.")

# ----------------------------------------------------------------------------
elif page.startswith("🎞️"):
    st.title("🎞️ Netflix Catalog Explorer")
    st.caption("8,807 Netflix titles · what does the catalog actually contain?")

    df = load_csv("netflix_titles.csv")

    kind = st.radio("Content type", ["All", "Movie", "TV Show"], horizontal=True)
    shown = df if kind == "All" else df[df["type"] == kind]

    years = shown["release_year"]
    lo, hi = st.slider("Release year range", int(years.min()), int(years.max()),
                       (2000, int(years.max())))
    shown = shown[(shown["release_year"] >= lo) & (shown["release_year"] <= hi)]

    st.metric("Titles in selection", f"{len(shown):,}")

    st.write("**Titles per release year:**")
    st.bar_chart(shown["release_year"].value_counts().sort_index())

    st.write("**Top 10 genres:**")
    genres = shown["listed_in"].str.split(", ").explode().value_counts().head(10)
    st.bar_chart(genres)

    st.write("**Top 10 producing countries:**")
    countries = shown["country"].dropna().str.split(", ").explode().value_counts().head(10)
    st.bar_chart(countries)

# ----------------------------------------------------------------------------
elif page.startswith("🏁"):
    st.title("🏁 Bar Chart Race: Country Populations")
    st.caption("Gapminder data animated with matplotlib — rendered in the notebook, replayed here")

    st.image("assets/bar_chart_race.gif")
    st.markdown("""
The notebook builds this animation frame by frame:

1. For each year, take the top-10 most populous countries
2. Draw a horizontal bar chart for that year
3. Stitch the frames into a GIF with `matplotlib.animation`

Animation is a *storytelling* tool — the same data in a static table
would never show China and India pulling away so vividly.
""")

# ----------------------------------------------------------------------------
elif page.startswith("🚕"):
    st.title("🚕 Uber Trips in New York (April 2014)")
    st.caption("60,000-trip sample · where and when does the city ride?")

    df = load_csv("uber_sample.csv", parse_dates=["Date/Time"])
    df["hour"] = df["Date/Time"].dt.hour
    df["weekday"] = df["Date/Time"].dt.day_name()

    hours = st.slider("Hour of day", 0, 23, (17, 20))
    sel = df[(df["hour"] >= hours[0]) & (df["hour"] <= hours[1])]

    st.metric("Trips in selection", f"{len(sel):,}")
    st.map(sel.sample(min(5000, len(sel)), random_state=1)
           .rename(columns={"Lat": "lat", "Lon": "lon"})[["lat", "lon"]])

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Trips by hour:**")
        st.bar_chart(df["hour"].value_counts().sort_index())
    with col2:
        st.write("**Trips by weekday:**")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        st.bar_chart(df["weekday"].value_counts().reindex(order))
    st.caption("Manhattan lights up during the evening rush — move the hour "
               "slider to watch the city's pulse.")

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

# ----------------------------------------------------------------------------
elif page.startswith("⭕"):
    st.title("⭕ Tic-Tac-Toe vs Minimax")
    st.caption("The agent searches every possible future — it has never lost a game")

    def winner(b):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
                 (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for i, j, k in lines:
            if b[i] != " " and b[i] == b[j] == b[k]:
                return b[i]
        return "draw" if " " not in b else None

    def minimax(b, player):
        w = winner(b)
        if w == "O":
            return 1, None
        if w == "X":
            return -1, None
        if w == "draw":
            return 0, None
        best_score = -2 if player == "O" else 2
        best_move = None
        for i in range(9):
            if b[i] == " ":
                b[i] = player
                score, _ = minimax(b, "X" if player == "O" else "O")
                b[i] = " "
                if (player == "O" and score > best_score) or (player == "X" and score < best_score):
                    best_score, best_move = score, i
        return best_score, best_move

    if "ttt" not in st.session_state:
        st.session_state.ttt = [" "] * 9

    board = st.session_state.ttt
    status = winner(board)

    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            i = r * 3 + c
            label = board[i] if board[i] != " " else "·"
            if cols[c].button(label, key=f"cell{i}", use_container_width=True,
                              disabled=(board[i] != " " or status is not None)):
                board[i] = "X"
                if winner(board) is None:
                    _, move = minimax(board, "O")
                    if move is not None:
                        board[move] = "O"
                st.rerun()

    status = winner(board)
    if status == "X":
        st.balloons()
        st.success("You beat minimax?! Check the code — that should be impossible 😄")
    elif status == "O":
        st.error("Minimax wins. As mathematically guaranteed.")
    elif status == "draw":
        st.info("Draw — the best possible outcome against a perfect player.")
    else:
        st.caption("You are ❌ — click a cell. The agent (⭕) replies instantly.")

    if st.button("🔄 New game"):
        st.session_state.ttt = [" "] * 9
        st.rerun()

# ----------------------------------------------------------------------------
elif page.startswith("🧭"):
    st.title("🧭 Q-Learning Gridworld")
    st.caption("Agent trained with 3,000 episodes of trial and error · walls, traps and a goal")

    agent = load_model("qlearning_agent")
    GRID, policy = agent["grid"], agent["policy"]
    ROWS, COLS = len(GRID), len(GRID[0])
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ARROWS = ["↑", "→", "↓", "←"]
    START, GOAL = (0, 0), (7, 7)

    # Greedy walk following the learned policy
    path, state, done, guard = [START], START, False, 0
    while not done and guard < 100:
        r, c = state
        dr, dc = ACTIONS[int(policy[r, c])]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS) or GRID[nr][nc] == "#":
            nr, nc = r, c
        state = (nr, nc)
        path.append(state)
        done = GRID[nr][nc] in "GX"
        guard += 1

    step_n = st.slider("Step through the agent's walk", 0, len(path) - 1, 0)
    show_policy = st.toggle("Show learned policy arrows")

    agent_pos = path[step_n]
    visited = set(path[:step_n + 1])

    rows_out = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = GRID[r][c]
            if (r, c) == agent_pos:
                row.append("🤖")
            elif cell == "#":
                row.append("⬛")
            elif cell == "G":
                row.append("🏁")
            elif cell == "X":
                row.append("💀")
            elif (r, c) in visited:
                row.append("🟨")
            elif show_policy:
                row.append(ARROWS[int(policy[r, c])])
            else:
                row.append("⬜")
        rows_out.append(row)

    st.table(pd.DataFrame(rows_out))
    if agent_pos == GOAL:
        st.success(f"Goal reached in {step_n} steps! 🏁")
    st.caption("⬛ wall · 💀 trap (-10) · 🏁 goal (+10) · every step costs -0.1, "
               "so the agent learned the *shortest safe* path — drag the slider to replay it.")
