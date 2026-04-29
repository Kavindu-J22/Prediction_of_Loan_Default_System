import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def load_and_sample_data(file_path, sample_size=50000, random_state=42):
    """
    Loads a random sample of the dataset to save memory and maps the target variable.
    Target: 0 = Fully Paid, 1 = Charged Off / Default.
    """
    # Define which columns we might definitely need or drop to save memory. 
    # For now, let's load everything but sample it.
    # To sample effectively from a large CSV without loading it all, we can read a chunk or 
    # load randomly. Given the constraints, we will read the CSV and sample. 
    # If the file is too large to pd.read_csv entirely, we can use an iterator.
    
    # Let's read a chunk to see if we can get enough valid target classes
    chunksize = 200000 
    sampled_df = pd.DataFrame()
    
    # We only want rows with Fully Paid, Charged Off, Default
    target_classes = ['Fully Paid', 'Charged Off', 'Default']
    
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        filtered_chunk = chunk[chunk['loan_status'].isin(target_classes)]
        sampled_df = pd.concat([sampled_df, filtered_chunk])
        if len(sampled_df) >= sample_size * 2: # Get more than needed to sample down later
            break
            
    # Sample down to exactly the sample_size
    if len(sampled_df) > sample_size:
        df = sampled_df.sample(n=sample_size, random_state=random_state).copy()
    else:
        df = sampled_df.copy()
        
    # Map target variable
    # 'Fully Paid' -> 0, 'Charged Off' -> 1, 'Default' -> 1
    target_mapping = {'Fully Paid': 0, 'Charged Off': 1, 'Default': 1}
    df['loan_status'] = df['loan_status'].map(target_mapping)
    
    # Reset index
    df = df.reset_index(drop=True)
    return df

def handle_missing_values(df, drop_threshold=0.5):
    """
    Drops columns with > drop_threshold missing values.
    Imputes remaining numerical with median, categorical with mode.
    """
    # 1. Drop columns with too many missing values
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df_cleaned = df.drop(columns=cols_to_drop).copy()
    
    # 2. Impute remaining missing values
    num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    
    # Numerical -> Median
    for col in num_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            
    # Categorical -> Mode
    for col in cat_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_val = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            
    return df_cleaned, cols_to_drop

def handle_outliers(df, method='cap', factor=1.5):
    """
    Handles outliers using the IQR method. 
    method: 'cap' (replace with upper/lower bounds) or 'drop' (remove rows).
    """
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=['int64', 'float64']).columns
    
    # Exclude target variable and IDs from outlier treatment
    cols_to_exclude = ['loan_status', 'id', 'member_id']
    num_cols = [c for c in num_cols if c not in cols_to_exclude]
    
    if method == 'drop':
        # Create a boolean mask for rows to keep
        keep_mask = np.ones(len(df_out), dtype=bool)
        for col in num_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            col_mask = (df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)
            keep_mask = keep_mask & col_mask
        df_out = df_out[keep_mask].reset_index(drop=True)
    
    elif method == 'cap':
        for col in num_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])
            
    return df_out

def encode_categorical(df, max_cardinality=15):
    """
    Encodes categorical features.
    Binary/Ordinal -> LabelEncoder
    Nominal -> One-Hot Encoding (pd.get_dummies)
    Drops columns with extremely high cardinality to avoid explosion.
    """
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object']).columns
    
    # Separate into binary, low cardinality, and high cardinality
    cols_to_drop = []
    
    for col in cat_cols:
        unique_count = df_encoded[col].nunique()
        
        if unique_count == 2:
            # Label encode
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        elif unique_count > max_cardinality:
            # High cardinality - drop for now (e.g. zip_code, url, emp_title)
            cols_to_drop.append(col)
        else:
            # Leave for One Hot Encoding
            pass
            
    # Drop high cardinality columns
    df_encoded = df_encoded.drop(columns=cols_to_drop)
    
    # Remaining object columns for One Hot Encoding
    ohe_cols = df_encoded.select_dtypes(include=['object']).columns
    if len(ohe_cols) > 0:
        df_encoded = pd.get_dummies(df_encoded, columns=ohe_cols, drop_first=True)
        
    return df_encoded, cols_to_drop

def scale_features(df, scaler_type='standard'):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.
    """
    df_scaled = df.copy()
    num_cols = df_scaled.select_dtypes(include=['int64', 'float64', 'uint8', 'int32']).columns
    
    # Exclude target and dummy categorical if they somehow get picked up, 
    # but mostly we want to scale real continuous variables.
    cols_to_exclude = ['loan_status', 'id', 'member_id']
    num_cols = [c for c in num_cols if c not in cols_to_exclude]
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    if len(num_cols) > 0:
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        
    return df_scaled
