# NLP Rating Prediction – Data Science Assignment  
Author: Rupa Manogna Vudumula

## 📌 Overview
This project predicts **Google Play Store app ratings (1–5)** from user reviews using **Natural Language Processing (NLP)** and Machine Learning.

The dataset contains:
- Review Text  
- Review Title  
- App Version information  
- Star Rating (target label for training)

The goal is to build an NLP model that achieves high **Weighted F1-score** on validation data and generate predictions for the test dataset.

---

## 📂 Project Structure

Data_science_rupa  
│── train.csv  
│── test.csv  
│── sample_submission.csv  
│── predictions.csv  
│── requirements.txt  
│── README.md  
└── src  
   │── data_preprocessing.py  
   │── train_model.py  
   └── predict.py  

---

## 🔧 Approach

### 1. **Text Preprocessing**
- Lowercasing  
- Removing special characters  
- Removing extra spaces  
- Cleaning both `Review Text` and `Review Title`  
- Combining them into a single text feature

### 2. **Feature Extraction**
## 🧠 2. Feature Extraction
Used **TF-IDF Vectorizer** with:

- analyzer="char_wb"
- ngram_range=(3,5)
- max_features=50000

This character-level TF-IDF setup captures subword patterns, 
handles typos, and improves performance for noisy review text. 
It is especially effective for sentiment-based text classification 
and works well for medium-sized datasets like this one.


### 3. **Model Training**
Two models were trained:

- **Logistic Regression (balanced class weights)**  
- **Linear SVM (balanced class weights)**  

Evaluation metric: **Weighted F1-Score**

The system automatically selects the **best-performing model**.

### 4. **Best Model**
- **Linear SVM**
- **Weighted F1 = 0.7263**

---

## 📈 Validation Strategy
- 80/20 train-validation split  
- Stratified sampling based on rating distribution  
- Weighted F1 used due to class imbalance (1–5 stars)

---

## 🚀 How to Run

### 1. Install requirements
pip install -r requirements.txt

### 2. Train the model
python src/train_model.py


### 3. Generate predictions


python src/predict.py


### 4. Output
A file named **`predictions.csv`** will be created.

---

## 📝 Final Notes
- The project follows the structure required in the assignment PDF.  
- The model achieves a strong balance between accuracy and F1-score.  
- This repository is fully ready for submission.
