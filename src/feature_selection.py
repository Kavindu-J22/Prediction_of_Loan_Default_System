import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def calculate_correlation(df, target_col='loan_status'):
    """
    Calculates the Pearson correlation of numerical features with the target column.
    Returns a dataframe sorted by absolute correlation.
    """
    # Select numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    if target_col not in num_cols:
        return pd.DataFrame()
        
    # Calculate correlation
    corr_matrix = df[num_cols].corr()
    
    # Extract correlation with target
    target_corr = corr_matrix[[target_col]].drop(index=[target_col, 'id', 'member_id'], errors='ignore')
    target_corr.columns = ['Correlation']
    target_corr['Abs_Correlation'] = target_corr['Correlation'].abs()
    
    # Sort by absolute correlation
    target_corr = target_corr.sort_values(by='Abs_Correlation', ascending=False)
    
    return target_corr

def calculate_chi_square(df, target_col='loan_status'):
    """
    Calculates Chi-Square statistic and p-value for categorical/binary features against the target.
    Returns a dataframe sorted by p-value (ascending).
    """
    results = []
    
    # Identify potential categorical/binary columns
    # In our pipeline, we have uint8 (from get_dummies) and small ints (Label Encoded)
    # Let's check for columns with a small number of unique values (e.g., < 100)
    for col in df.columns:
        if col in [target_col, 'id', 'member_id']:
            continue
            
        unique_vals = df[col].nunique()
        # Treat as categorical for Chi-Square if it has few unique values OR is uint8/bool
        if unique_vals < 50 or df[col].dtype == 'uint8' or df[col].dtype == 'bool':
            # Create contingency table
            contingency_table = pd.crosstab(df[col], df[target_col])
            
            # Ensure the table is at least 2x2
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                try:
                    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                    results.append({
                        'Feature': col,
                        'Chi-Square Statistic': chi2,
                        'p-value': p_val,
                        'Unique Values': unique_vals
                    })
                except ValueError:
                    pass # Skip if table contains 0s that cause issues, though chi2_contingency usually handles it
                    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='p-value', ascending=True)
        
    return results_df
