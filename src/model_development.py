import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col='loan_status', features_to_drop=None):
    """
    Prepares the data by separating features and target, and dropping excluded features.
    Returns X_train, X_test, y_train, y_test.
    """
    if features_to_drop is None:
        features_to_drop = []
        
    drop_cols = [target_col, 'id', 'member_id'] + features_to_drop
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    
    # Sanitize feature names for XGBoost / LightGBM
    X.columns = X.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    
    y = df[target_col]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """
    Generates predictions and calculates core metrics.
    """
    start_time = time.time()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    
    end_time = time.time()
    
    results = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "Inference Time (s)": round(end_time - start_time, 4)
    }
    
    return results

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression baseline model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier.
    """
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def train_lightgbm(X_train, y_train):
    """
    Trains a LightGBM classifier.
    """
    model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def get_feature_importance(model, feature_names):
    """
    Extracts feature importances from tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=False)
        return df
    elif hasattr(model, 'coef_'):
        # For Logistic Regression
        importances = np.abs(model.coef_[0])
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=False)
        return df
    else:
        return pd.DataFrame()
