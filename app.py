import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any, List

# --- Configuration ---
CSV_FILE = 'events.csv' # Assuming this file exists in the same directory
RANDOM_SEED = 42
INTERACTION_THRESHOLD = 3
TOP_K_DEFAULT = 10

# --- Helper Functions (Reorganized Notebook Logic) ---

@st.cache_data
def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[int, Any], Dict[int, Any], np.ndarray]:
    """
    Loads, cleans, preprocesses the data, and performs negative sampling.
    Returns: Cleaned DataFrame, features X, labels y, index-to-original-ID mappings, and the full encoded features X_all.
    """
    st.info(f"Loading and preparing data from **{file_path}**...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
        st.stop()
        
    original_shape = df.shape
    
    # 2. Data Cleaning & Preparation
    
    # Drop rows with missing core fields
    df.dropna(subset=['timestamp', 'visitorid', 'event', 'itemid'], inplace=True)
    
    # Convert timestamp to datetime (if not already)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # 3. Filtering
    
    # Filter out visitors and items with fewer than 3 interactions.
    visitor_counts = df['visitorid'].value_counts()
    item_counts = df['itemid'].value_counts()
    
    valid_visitors = visitor_counts[visitor_counts >= INTERACTION_THRESHOLD].index
    valid_items = item_counts[item_counts >= INTERACTION_THRESHOLD].index
    
    df = df[df['visitorid'].isin(valid_visitors)]
    df = df[df['itemid'].isin(valid_items)]
    
    st.info(f"Data reduced from {original_shape} rows to {df.shape} rows after cleaning and filtering.")

    # 4. Create Binary Label
    df['rating'] = df['event'].apply(
        lambda x: 1 if x in ['addtocart', 'transaction'] else 0
    )

    # 5. User-aware Negative Sampling
    
    # Keep all positives (rating=1)
    df_positives = df[df['rating'] == 1].copy()
    
    # Sample negatives (rating=0) per visitor
    df_negatives = df[df['rating'] == 0].copy()
    
    # Use the number of unique positive interactions as a guide for balanced sampling.
    # A simple approach is to ensure the number of negatives is roughly equal to positives
    # or to sample based on a visitor's activity.
    
    # For simplicity and to match the 'balanced dataset' intent:
    # We'll sample negatives to keep the final dataset close to balanced,
    # prioritizing keeping all unique positive (visitor, item) pairs.
    
    # Get all unique (visitor, item) pairs for negatives
    unique_negative_pairs = df_negatives[['visitorid', 'itemid']].drop_duplicates()
    
    # The final goal is to model the unique interaction pair (visitorid, itemid)
    # So we should group and consolidate data to unique (visitor, item) pairs first.
    # The highest rating for a pair (1 if any positive, 0 otherwise) is the label.
    
    df_interactions = df.groupby(['visitorid', 'itemid'], as_index=False)['rating'].max()
    
    # Separate consolidated positives and negatives
    df_positives_final = df_interactions[df_interactions['rating'] == 1]
    df_negatives_final = df_interactions[df_interactions['rating'] == 0]
    
    # Sample negative interactions to match the number of positives
    num_positives = len(df_positives_final)
    if len(df_negatives_final) > num_positives:
        df_negatives_sampled = df_negatives_final.sample(n=num_positives, random_state=RANDOM_SEED)
    else:
        df_negatives_sampled = df_negatives_final # Keep all if fewer than positives
        
    df_balanced = pd.concat([df_positives_final, df_negatives_sampled]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    st.info(f"Balanced dataset created with {len(df_balanced)} unique (visitor, item) interactions (approx. 50% positive).")
    
    # 6. Encoding
    
    # Encode visitorid and itemid as categorical integer codes
    df_balanced['user_id_enc'] = df_balanced['visitorid'].astype('category').cat.codes
    df_balanced['item_id_enc'] = df_balanced['itemid'].astype('category').cat.codes
    
    # Create Mappings
    visitor_map = df_balanced[['user_id_enc', 'visitorid']].drop_duplicates().set_index('user_id_enc')['visitorid']
    item_map = df_balanced[['item_id_enc', 'itemid']].drop_duplicates().set_index('item_id_enc')['itemid']
    
    idx_to_visitor = visitor_map.to_dict()
    idx_to_item = item_map.to_dict()
    
    # 7. Build X and y
    X = df_balanced[['user_id_enc', 'item_id_enc']].values
    y = df_balanced['rating'].values
    
    # X_all (for recommendation: the full set of encoded interactions)
    # This is the features matrix for ALL interactions used to train the model.
    X_all = X.copy() 

    return df_balanced, X, y, idx_to_visitor, idx_to_item, X_all


@st.cache_resource(show_spinner="Training Recommendation Model...")
def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, Dict[int, Any], Dict[int, Any], np.ndarray]:
    """
    Splits data, creates a scikit-learn Pipeline with OneHotEncoder and Logistic Regression,
    trains the model, and returns it along with the mappings and the full feature matrix.
    
    Note: The pipeline will be trained on the encoded IDs.
    """
    
    # Split into train/test (required for evaluation in the notebook, though not strictly for the app)
    # We will train the final model on X_train only, but in a real-world scenario, you might retrain on X_all.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # 1. Create Column Transformer for OneHotEncoding
    # Columns 0 and 1 are 'user_id_enc' and 'item_id_enc'
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot_user', OneHotEncoder(handle_unknown="ignore"), [0]),
            ('onehot_item', OneHotEncoder(handle_unknown="ignore"), [1])
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    
    # 2. Create Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000, 
            class_weight="balanced", 
            n_jobs=-1, 
            random_state=RANDOM_SEED,
            solver='liblinear' # 'lbfgs' is default, but 'liblinear' can be faster for small datasets
        ))
    ])
    
    # 3. Train Model
    st.info(f"Training Logistic Regression Pipeline on {len(X_train)} samples...")
    model_pipeline.fit(X_train, y_train)
    st.success("Model training complete!")
    
    # In a real app, we usually cache the model trained on the FULL dataset (X, y)
    # To follow the prompt exactly and avoid retraining, we will return the trained model.
    # For a *final* deployment, fit on (X, y) before caching.
    
    # We will return the model trained on the full set to maximize recommendation quality
    st.info("Re-training model on full balanced dataset for better recommendations...")
    model_pipeline.fit(X, y)
    
    return model_pipeline


def recommend_items_for_user_encoded(
    model: Pipeline, 
    user_idx: int, 
    X_all: np.ndarray, 
    top_k: int
) -> pd.DataFrame:
    """
    Finds unseen items for a user, scores them, and returns the Top-K with scores.
    """
    
    # 1. Find all unique item IDs
    all_item_ids = np.unique(X_all[:, 1])
    
    # 2. Find items the user has ALREADY interacted with (seen)
    user_interactions = X_all[X_all[:, 0] == user_idx]
    seen_item_ids = np.unique(user_interactions[:, 1])
    
    # 3. Find unseen items
    unseen_item_ids = np.setdiff1d(all_item_ids, seen_item_ids)
    
    if len(unseen_item_ids) == 0:
        return pd.DataFrame() # Return empty DataFrame if no unseen items
    
    # 4. Create feature matrix for prediction
    user_idx_array = np.full((len(unseen_item_ids), 1), user_idx)
    X_predict = np.hstack([user_idx_array, unseen_item_ids.reshape(-1, 1)])
    
    # 5. Score items (predict_proba returns [P(class 0), P(class 1)])
    probabilities = model.predict_proba(X_predict)[:, 1] # Probability of positive class (rating=1)
    
    # 6. Rank and select Top-K
    results = pd.DataFrame({
        'encoded_user_id': user_idx,
        'encoded_item_id': unseen_item_ids,
        'score': probabilities
    })
    
    results = results.sort_values(by='score', ascending=False).head(top_k)
    
    return results.reset_index(drop=True)


# --- Streamlit App ---

st.set_page_config(
    page_title="E-Commerce Recommendation Engine",
    layout="wide"
)

def main():
    """Main function to run the Streamlit application."""
    
    st.title("🛒 E-Commerce Recommendation Engine (Logistic Regression)")
    st.markdown("This app demonstrates a simple recommendation engine based on a Logistic Regression model trained on e-commerce interaction data. The model predicts the probability that a user will have a positive interaction (add to cart/transaction) with an item they haven't seen before.")
    
    st.sidebar.header("⚙️ Data & Model Setup")
    
    # --- Load Data and Train Model (Cached) ---
    
    # 1. Load Data
    with st.spinner("Loading and preprocessing data..."):
        try:
            df_balanced, X, y, idx_to_visitor, idx_to_item, X_all = load_and_prepare_data(CSV_FILE)
        except Exception as e:
            # Error is handled in the data loading function, but catch generic errors here too
            st.error(f"Failed to load or prepare data: {e}")
            return # Stop execution
            
    # 2. Train Model
    # The return includes the full feature matrix X_all and mappings.
    model = train_model(X, y)
    
    # --- UI Components ---

    # Get range for validation
    min_user_id = X_all[:, 0].min()
    max_user_id = X_all[:, 0].max()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recommendation Parameters")

    # Input Box for User ID
    user_idx_input = st.sidebar.number_input(
        f"Enter Encoded User ID (Integer):",
        min_value=int(min_user_id),
        max_value=int(max_user_id),
        value=int(min_user_id), # Default to the first user
        step=1,
        format="%d",
        key="user_id_input"
    )
    
    # Display valid range
    st.sidebar.markdown(f"**Valid Range:** `{int(min_user_id)}` to `{int(max_user_id)}`")

    # Slider for Top K
    top_k_value = st.sidebar.slider(
        "Select Top K Recommendations:",
        min_value=1,
        max_value=20,
        value=TOP_K_DEFAULT,
        step=1
    )
    
    st.sidebar.markdown("---")
    recommend_button = st.sidebar.button("Get Recommendations 🚀", type="primary")

    # --- Recommendation Logic ---
    
    if recommend_button:
        st.subheader(f"Recommendations for Encoded User ID: `{user_idx_input}`")
        
        # 1. Validation
        if user_idx_input not in idx_to_visitor:
            st.error(f"Error: Encoded User ID **{user_idx_input}** does not exist in the training data.")
            return

        # 2. Get Recommendations
        with st.spinner(f"Finding top {top_k_value} unseen items..."):
            recommendations_df = recommend_items_for_user_encoded(
                model, 
                user_idx_input, 
                X_all, 
                top_k_value
            )
        
        # 3. Display Results
        if recommendations_df.empty:
            st.warning(f"No unseen items found for Encoded User ID **{user_idx_input}**. This user may have interacted with all available items.")
        else:
            # Map encoded IDs back to original IDs
            recommendations_df['visitorid'] = recommendations_df['encoded_user_id'].map(idx_to_visitor)
            recommendations_df['itemid'] = recommendations_df['encoded_item_id'].map(idx_to_item)
            
            # Reorder and format columns
            final_cols = ['visitorid', 'itemid', 'encoded_user_id', 'encoded_item_id', 'score']
            recommendations_df = recommendations_df[final_cols]
            
            # Format the score as a percentage
            recommendations_df['score'] = (recommendations_df['score'] * 100).map('{:.2f}%'.format)
            
            st.dataframe(recommendations_df, use_container_width=True)
            
            st.success(f"Successfully generated {len(recommendations_df)} recommendations.")

if __name__ == "__main__":
    main()