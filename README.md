

# **📌 TransferIQ: Dynamic Player Transfer Value Prediction using AI & Multi-Source Data**

## **📖 Overview**

TransferIQ is an AI-powered system designed to predict football player transfer values using multiple data sources such as player performance statistics, injury history, social media sentiment, and historical market value data. The project combines time-series deep learning models (LSTM) with advanced ensemble models (XGBoost) to generate accurate and dynamic transfer value predictions.

---

## **🎯 Project Objectives**

* Predict player transfer values using multi-source, real-world football data
* Analyze factors like performance, age, injuries, and sentiment
* Build LSTM, multivariate LSTM, and encoder-decoder forecasting models
* Develop ensemble models combining LSTM and XGBoost
* Visualize trends and generate insights for sports analytics decision-making

---

## **📂 Datasets Used**

* **StatsBomb Performance Data** – passes, shots, xG, dribbles, defensive stats
* **Kaggle Player Transfer Value Dataset**
* **Injury Dataset** – injury type, duration, recovery period
* **Twitter API Sentiment Data** – preprocessed using VADER/TextBlob

---

## **🔧 Key Technologies**

* Python, Pandas, NumPy
* TensorFlow & Keras (LSTM models)
* XGBoost (Ensemble learning)
* NLP Tools (VADER, TextBlob)
* Matplotlib, Seaborn, Plotly

---

## **🧠 Model Pipeline**

1. **Data Collection & Cleaning**
2. **Feature Engineering**

   * Injury risk score
   * Performance trend features
   * Sentiment score
3. **Time-Series Modeling**

   * Univariate LSTM
   * Multivariate LSTM
   * Encoder–Decoder LSTM
4. **Ensemble Modeling with XGBoost**
5. **Evaluation Using RMSE, MAE, R²**
6. **Visualization & Insights**

---

## **📊 Final Outputs**

* Predicted transfer values
* Player performance trend charts
* Injury impact analysis
* Sentiment influence on market value
* Feature importance and model comparison

---

## **⭐ Key Insights**

* Younger players with strong performance and positive sentiment show higher value growth
* Injury frequency significantly reduces market value
* Ensemble model produced the most stable and accurate predictions

---

## **📘 Conclusion**

TransferIQ demonstrates how AI and multi-source data can enhance player valuation in football. This project improved skills in data processing, NLP, time-series modeling, and ensemble learning, providing a real-world understanding of sports analytics applications.


