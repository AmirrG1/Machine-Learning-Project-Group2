# E-Commerce Recommendation Engine (Machine Learning Project)

This repository contains our EECE454 Machine Learning course project.  
We build a **Top-K recommendation engine** for an e-commerce website using **Logistic Regression** (no neural networks).

---

## Files

- `E-Commerce Recommendation Engine - ML - FINAL.ipynb`  
  Final Jupyter notebook with all preprocessing, model training, and evaluation.

- `events.zip`  
  Compressed version of the dataset (`events.csv`). Unzip it in this folder before running the notebook or app.

- `app.py`  
  Streamlit app that lets you interact with the recommender.

---

## Dataset

The dataset (`events.csv`) contains user–item interactions with columns:

- `timestamp`
- `visitorid`
- `event` (`view`, `addtocart`, `transaction`)
- `itemid`
- `transactionid`

We create a binary label:

- `rating = 1` for `addtocart` or `transaction`
- `rating = 0` for `view`

We then filter cold users/items and do user-aware negative sampling to get a more balanced training set.

---

## Model

We treat recommendation as a **binary classification** problem:

> Will this user have a positive interaction with this item (1) or not (0)?

Model:

- `OneHotEncoder(handle_unknown="ignore")` on user and item IDs  
- `LogisticRegression(max_iter=1000, class_weight="balanced")`  
- Wrapped in a scikit-learn `Pipeline`

The model outputs a **probability**, which we use to **rank items** and compute metrics like **Precision@10** and **Recall@10**.

---

## How to Run the Notebook

1. Clone the repo:
   git clone https://github.com/AmirrG1/Machine-Learning-Project-Group2.git
   cd Machine-Learning-Project-Group2
Unzip the dataset:

unzip events.zip   # creates events.csv
Launch Jupyter and open:


E-Commerce Recommendation Engine - ML - FINAL.ipynb
Run all cells in order.

How to Run the Streamlit App
Install dependencies (example):


pip install streamlit scikit-learn pandas numpy
From the project folder:


streamlit run app.py
Open the local URL shown in the terminal (usually http://localhost:8501).

Group Members
Amirali Tamizi Horzadeh (Github username: @AmirrG1) 2105035420
Ahmad Alsayegh 2112036252
Aly Mabrouk 2208037712
Saeed Alshehhi 2108035865
