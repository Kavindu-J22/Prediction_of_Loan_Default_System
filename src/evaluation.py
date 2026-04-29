import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

def get_roc_curve_data(model, X_test, y_test, model_name):
    """
    Returns fpr, tpr, and auc for a given model.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if probability is not supported (not expected for our chosen models)
        y_prob = model.predict(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, model_name

def get_pr_curve_data(model, X_test, y_test, model_name):
    """
    Returns precision, recall, and auc-pr for a given model.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
        
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    return precision, recall, pr_auc, model_name

def get_confusion_matrix_data(model, X_test, y_test):
    """
    Returns the confusion matrix.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm
