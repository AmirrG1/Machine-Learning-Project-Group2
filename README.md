# E-Commerce Recommendation Engine

A machine learning recommendation system that ranks e-commerce items for users based on historical user–item interactions. The project treats recommendation as a binary classification problem and uses Logistic Regression to predict whether a user is likely to interact positively with an item.

## Project Overview

This project builds a **Top-K recommendation engine** for an e-commerce dataset using Python and Scikit-learn. The model predicts the probability of a positive interaction between a user and an item, then ranks items to generate recommendations.

Positive interactions are defined as:

* `addtocart`
* `transaction`

Negative interactions are defined as:

* `view`

## Key Results

* **Precision@10:** 0.64
* **Recall@10:** 0.98
* Built a complete ML pipeline covering preprocessing, user/item filtering, negative sampling, model training, evaluation, and interactive deployment through Streamlit.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Logistic Regression
* OneHotEncoder
* Streamlit
* Jupyter Notebook

## Dataset

The dataset contains user–item event interactions with the following columns:

* `timestamp`
* `visitorid`
* `event`
* `itemid`
* `transactionid`

A binary label was created:

* `rating = 1` for `addtocart` or `transaction`
* `rating = 0` for `view`

To improve training quality, cold users/items were filtered and user-aware negative sampling was applied to create a more balanced dataset.

## Model Approach

The recommendation problem was treated as a binary classification task:

> Will this user have a positive interaction with this item?

The pipeline includes:

* One-hot encoding of user and item IDs
* Logistic Regression with balanced class weights
* Probability-based ranking of items
* Top-K evaluation using Precision@10 and Recall@10

## Repository Files

```text
E-Commerce Recommendation Engine - ML - FINAL.ipynb
```

Final Jupyter notebook containing preprocessing, model training, and evaluation.

```text
events.zip
```

Compressed dataset file. Unzip it before running the notebook or app.

```text
app.py
```

Streamlit application for interacting with the recommendation engine.

## How to Run

Clone the repository:

```bash
git clone https://github.com/AmirrG1/ecommerce-recommendation-engine.git
cd ecommerce-recommendation-engine
```

Unzip the dataset:

```bash
unzip events.zip
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn streamlit
```

Run the notebook:

```bash
jupyter notebook
```

Open:

```text
E-Commerce Recommendation Engine - ML - FINAL.ipynb
```

Run all cells in order.

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## What I Learned

* Building an end-to-end machine learning pipeline
* Preparing user–item interaction data for recommendation tasks
* Applying negative sampling to improve training balance
* Using Logistic Regression for ranking-based recommendation
* Evaluating recommender systems with Precision@K and Recall@K
* Deploying an interactive ML project using Streamlit

## Author

Amirali Tamizi Horzadeh
GitHub: [@AmirrG1](https://github.com/AmirrG1)
