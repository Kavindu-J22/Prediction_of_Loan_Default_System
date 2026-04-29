import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import (
    load_and_sample_data,
    handle_missing_values,
    handle_outliers,
    encode_categorical,
    scale_features
)
from src.feature_selection import (
    calculate_correlation,
    calculate_chi_square
)
from src.model_development import (
    prepare_data,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    get_feature_importance
)
from src.evaluation import (
    get_roc_curve_data,
    get_pr_curve_data,
    get_confusion_matrix_data
)

# Set page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.title("💳 Loan Default Prediction Pipeline")

# ==========================================
# Sidebar Navigation & Settings
# ==========================================
st.sidebar.title("Navigation")
current_step = st.sidebar.radio("Go to Step:", [
    "1. Data Preprocessing", 
    "2. Feature Selection",
    "3. Model Development",
    "4. Rigorous Evaluation"
])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration Parameters")

# 1. Sampling Size
sample_size = st.sidebar.slider(
    "1. Data Sample Size",
    min_value=10000, max_value=200000, value=50000, step=10000,
    help="Number of records to sample from the dataset."
)

# ==========================================
# Data Loading (Cached)
# ==========================================
@st.cache_data(show_spinner="Loading and sampling dataset... This might take a minute.")
def load_data(sample_size):
    return load_and_sample_data('Dataset/loan_Dataset.csv', sample_size=sample_size)

try:
    raw_df = load_data(sample_size)
    st.sidebar.success(f"Successfully loaded {len(raw_df)} records.")
except Exception as e:
    st.error(f"Error loading dataset. Please ensure 'Dataset/loan_Dataset.csv' exists. Details: {e}")
    st.stop()


# ==========================================
# STEP 1: DATA PREPROCESSING
# ==========================================
if current_step == "1. Data Preprocessing":
    st.subheader("Step 1: Data Preprocessing (Cleaning & Shaping)")
    
    missing_threshold = st.sidebar.slider("Missing Value Drop Threshold", 0.1, 0.9, 0.5, 0.05)
    outlier_method = st.sidebar.selectbox("Outlier Handling Method (IQR)", ["cap", "drop", "none"], index=0)
    scaler_type = st.sidebar.selectbox("Feature Scaling Method", ["standard", "minmax", "none"], index=0)
    
    # Executing Pipeline
    df_clean, dropped_cols = handle_missing_values(raw_df, drop_threshold=missing_threshold)
    df_outliers = handle_outliers(df_clean, method=outlier_method) if outlier_method != "none" else df_clean.copy()
    df_encoded, high_card_cols_dropped = encode_categorical(df_outliers)
    df_final = scale_features(df_encoded, scaler_type=scaler_type) if scaler_type != "none" else df_encoded.copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Overview & Target", 
        "2. Missing Values", 
        "3. Outliers", 
        "4. Encoding", 
        "5. Final Scaled Data"
    ])

    with tab1:
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows Sampled", f"{raw_df.shape[0]:,}")
        col2.metric("Total Columns", f"{raw_df.shape[1]}")
        col3.metric("Memory Usage", f"{raw_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        target_counts = raw_df['loan_status'].value_counts().reset_index()
        target_counts.columns = ['Loan Status', 'Count']
        target_counts['Loan Status'] = target_counts['Loan Status'].map({0: '0 (Fully Paid)', 1: '1 (Default)'})
        fig = px.bar(target_counts, x='Loan Status', y='Count', color='Loan Status', title="Target Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(raw_df.head(10))

    with tab2:
        st.info(f"Dropped {len(dropped_cols)} columns due to > {missing_threshold*100}% missing values.")
        missing_before = raw_df.isnull().mean() * 100
        missing_before = missing_before[missing_before > 0].sort_values(ascending=False).head(20).reset_index()
        missing_before.columns = ['Feature', '% Missing']
        if len(missing_before) > 0:
            fig2 = px.bar(missing_before, x='Feature', y='% Missing', title="Top 20 Features with Missing Values")
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if outlier_method == "drop":
            st.warning(f"Rows dropped due to outliers: {len(df_clean) - len(df_outliers)}")
        num_cols_for_plot = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in ['id', 'member_id', 'loan_status']:
            if col in num_cols_for_plot: num_cols_for_plot.remove(col)
        
        if num_cols_for_plot:
            selected_col = st.selectbox("Select numerical feature:", num_cols_for_plot)
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(px.box(df_clean, y=selected_col, title="Before"), use_container_width=True)
            with c2: st.plotly_chart(px.box(df_outliers, y=selected_col, title="After"), use_container_width=True)

    with tab4:
        st.info(f"Dropped {len(high_card_cols_dropped)} high cardinality features.")
        st.metric("New Column Count", df_encoded.shape[1])
        st.dataframe(df_encoded.head(10))

    with tab5:
        st.dataframe(df_final.head(10))
        st.success("Pipeline Completed Successfully!")

# ==========================================
# STEP 2: FEATURE SELECTION
# ==========================================
elif current_step == "2. Feature Selection":
    st.subheader("Step 2: Feature Selection (Identifying Key Drivers)")
    
    with st.spinner("Ensuring preprocessing pipeline is up to date..."):
        df_clean, _ = handle_missing_values(raw_df, drop_threshold=0.5)
        df_outliers = handle_outliers(df_clean, method='cap')
        df_encoded, _ = encode_categorical(df_outliers)
        df_final = scale_features(df_encoded, scaler_type='standard')
        
    st.markdown("""
    This step identifies the most relevant attributes influencing Loan Default using statistical methods.
    - **Numerical Features:** Pearson Correlation Analysis
    - **Categorical Features:** Chi-Square Tests of Independence
    """)

    tab_corr, tab_chi, tab_final = st.tabs(["1. Correlation Analysis", "2. Chi-Square Tests", "3. Final Features"])
    
    with tab_corr:
        st.header("Correlation Analysis (Numerical Features)")
        corr_df = calculate_correlation(df_final)
        
        if not corr_df.empty:
            top_n = st.slider("Select Top N Correlated Features", min_value=5, max_value=50, value=20)
            top_corr = corr_df.head(top_n).reset_index()
            top_corr.columns = ['Feature', 'Correlation', 'Abs_Correlation']
            
            fig_corr = px.bar(top_corr, x='Correlation', y='Feature', orientation='h',
                              color='Correlation', color_continuous_scale='RdBu_r',
                              title=f"Top {top_n} Correlated Features with Loan Status")
            fig_corr.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_corr, use_container_width=True)
            st.dataframe(corr_df)
        else:
            st.warning("No numerical features available.")

    with tab_chi:
        st.header("Chi-Square Test (Categorical/Binary Features)")
        with st.spinner("Calculating Chi-Square statistics..."):
            chi_df = calculate_chi_square(df_encoded)
            
        if not chi_df.empty:
            st.markdown("**Features with p-value < 0.05 are statistically significant.**")
            def highlight_pval(val):
                color = '#c6efce' if val < 0.05 else '#ffc7ce'
                return f'background-color: {color}'
            st.dataframe(chi_df.style.map(highlight_pval, subset=['p-value']))
        else:
            st.warning("No categorical features available.")

    with tab_final:
        st.header("Feature Set Refinement")
        
        all_features = df_final.columns.tolist()
        for col in ['loan_status', 'id', 'member_id']:
            if col in all_features: all_features.remove(col)
        
        st.write(f"**Total Features Currently Available:** {len(all_features)}")
        
        features_to_drop = st.multiselect(
            "Select features to EXCLUDE from the final dataset (Optional):",
            options=all_features,
            key='features_to_drop'
        )
        
        final_selected_features = [f for f in all_features if f not in features_to_drop]
        st.success(f"**{len(final_selected_features)}** features retained for Model Development (Step 3).")

# ==========================================
# STEP 3: MODEL DEVELOPMENT
# ==========================================
elif current_step == "3. Model Development":
    st.subheader("Step 3: Model Development & Benchmarking")
    st.markdown("""
    Compare the traditional Econometric baseline (Logistic Regression) against Advanced Machine Learning models.
    All models are trained on the exact same preprocessed data to ensure a consistent framework.
    """)
    
    features_to_drop = st.session_state.get('features_to_drop', [])
    
    with st.spinner("Preparing Data for Modeling..."):
        df_clean, _ = handle_missing_values(raw_df, drop_threshold=0.5)
        df_outliers = handle_outliers(df_clean, method='cap')
        df_encoded, _ = encode_categorical(df_outliers)
        df_final = scale_features(df_encoded, scaler_type='standard')
        
        X_train, X_test, y_train, y_test = prepare_data(df_final, target_col='loan_status', features_to_drop=features_to_drop)
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        
    st.info(f"**Training Set:** {X_train.shape[0]:,} rows | **Test Set:** {X_test.shape[0]:,} rows | **Features:** {X_train.shape[1]}")

    tab_train, tab_perf, tab_feat_imp = st.tabs(["1. Train Models", "2. Performance Snapshot", "3. Feature Importance"])
    
    with tab_train:
        st.header("Select & Train Models")
        
        col1, col2, col3, col4 = st.columns(4)
        run_logreg = col1.checkbox("Logistic Regression (Baseline)", value=True)
        run_rf = col2.checkbox("Random Forest", value=True)
        run_xgb = col3.checkbox("XGBoost", value=True)
        run_lgbm = col4.checkbox("LightGBM", value=True)
        
        if st.button("🚀 Start Training", type="primary"):
            st.session_state['model_results'] = []
            st.session_state['trained_models'] = {}
            st.session_state['training_complete'] = False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_models = sum([run_logreg, run_rf, run_xgb, run_lgbm])
            if total_models == 0:
                st.warning("Please select at least one model to train.")
            else:
                current_model_idx = 0
                
                if run_logreg:
                    status_text.text("Training Logistic Regression...")
                    model_lr, t_time = train_logistic_regression(X_train, y_train)
                    res = evaluate_model(model_lr, X_test, y_test, "Logistic Regression")
                    res['Training Time (s)'] = round(t_time, 4)
                    st.session_state['model_results'].append(res)
                    st.session_state['trained_models']['Logistic Regression'] = model_lr
                    current_model_idx += 1
                    progress_bar.progress(current_model_idx / total_models)
                    
                if run_rf:
                    status_text.text("Training Random Forest...")
                    model_rf, t_time = train_random_forest(X_train, y_train)
                    res = evaluate_model(model_rf, X_test, y_test, "Random Forest")
                    res['Training Time (s)'] = round(t_time, 4)
                    st.session_state['model_results'].append(res)
                    st.session_state['trained_models']['Random Forest'] = model_rf
                    current_model_idx += 1
                    progress_bar.progress(current_model_idx / total_models)
                    
                if run_xgb:
                    status_text.text("Training XGBoost...")
                    model_xgb, t_time = train_xgboost(X_train, y_train)
                    res = evaluate_model(model_xgb, X_test, y_test, "XGBoost")
                    res['Training Time (s)'] = round(t_time, 4)
                    st.session_state['model_results'].append(res)
                    st.session_state['trained_models']['XGBoost'] = model_xgb
                    current_model_idx += 1
                    progress_bar.progress(current_model_idx / total_models)
                    
                if run_lgbm:
                    status_text.text("Training LightGBM...")
                    model_lgb, t_time = train_lightgbm(X_train, y_train)
                    res = evaluate_model(model_lgb, X_test, y_test, "LightGBM")
                    res['Training Time (s)'] = round(t_time, 4)
                    st.session_state['model_results'].append(res)
                    st.session_state['trained_models']['LightGBM'] = model_lgb
                    current_model_idx += 1
                    progress_bar.progress(current_model_idx / total_models)
                    
                status_text.text("Training Complete!")
                st.session_state['training_complete'] = True
                st.success("Models trained successfully! Move to the next tabs or Step 4 for evaluation.")

    with tab_perf:
        st.header("Initial Performance Benchmark")
        if st.session_state.get('training_complete', False):
            results_df = pd.DataFrame(st.session_state['model_results'])
            cols = ['Model', 'Accuracy', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Time (s)']
            results_df = results_df[cols]
            
            def highlight_max(s, props=''):
                return np.where(s == np.nanmax(s.values), props, '')
                
            st.dataframe(results_df.style.apply(highlight_max, props='background-color:#c6efce;', subset=['Accuracy', 'AUC-ROC', 'F1-Score']))
            
            fig = px.bar(results_df, x='Model', y='AUC-ROC', color='Model', title="AUC-ROC Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train models in the first tab to view performance metrics.")

    with tab_feat_imp:
        st.header("Model Introspection: Feature Importance")
        if st.session_state.get('training_complete', False):
            trained_models = st.session_state['trained_models']
            selected_model_name = st.selectbox("Select Model to View Importances:", list(trained_models.keys()))
            
            model = trained_models[selected_model_name]
            imp_df = get_feature_importance(model, st.session_state['X_test'].columns)
            
            if not imp_df.empty:
                top_n = st.slider("Top N Features", 5, 30, 15)
                fig_imp = px.bar(imp_df.head(top_n), x='Importance', y='Feature', orientation='h',
                                 title=f"Top {top_n} Features - {selected_model_name}",
                                 color='Importance', color_continuous_scale='Viridis')
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Feature importance is not available for this model configuration.")
        else:
            st.info("Train models to view feature importances.")

# ==========================================
# STEP 4: RIGOROUS EVALUATION
# ==========================================
elif current_step == "4. Rigorous Evaluation":
    st.subheader("Step 4: Rigorous Evaluation")
    
    if not st.session_state.get('training_complete', False):
        st.warning("⚠️ You must train the models in **Step 3** before evaluating them here.")
    else:
        st.markdown("""
        Compare the models using deep statistical visualizations to explicitly measure the trade-offs between 
        True Positives (catching defaults) and False Positives (denying good loans).
        """)
        
        trained_models = st.session_state['trained_models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        tab_roc, tab_pr, tab_cm, tab_metrics = st.tabs([
            "1. ROC Curves", 
            "2. Precision-Recall", 
            "3. Confusion Matrices", 
            "4. Full Metrics"
        ])
        
        with tab_roc:
            st.header("Receiver Operating Characteristic (ROC) Curves")
            st.markdown("Shows how well the models distinguish between classes. A curve closer to the top-left corner indicates higher AUC.")
            
            fig_roc = go.Figure()
            for name, model in trained_models.items():
                fpr, tpr, roc_auc, _ = get_roc_curve_data(model, X_test, y_test, name)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {roc_auc:.3f})'))
                
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash', color='grey')))
            fig_roc.update_layout(title='Multi-Model ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with tab_pr:
            st.header("Precision-Recall Curves")
            st.markdown("Particularly useful for imbalanced datasets like Loan Defaults. Higher area under the curve is better.")
            
            fig_pr = go.Figure()
            for name, model in trained_models.items():
                precision, recall, pr_auc, _ = get_pr_curve_data(model, X_test, y_test, name)
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{name} (AUC-PR = {pr_auc:.3f})'))
                
            fig_pr.update_layout(title='Multi-Model Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision')
            st.plotly_chart(fig_pr, use_container_width=True)
            
        with tab_cm:
            st.header("Confusion Matrices")
            st.markdown("Visually compare the True Positives, True Negatives, False Positives, and False Negatives.")
            
            cols = st.columns(len(trained_models))
            
            for i, (name, model) in enumerate(trained_models.items()):
                cm = get_confusion_matrix_data(model, X_test, y_test)
                
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Fully Paid (0)', 'Default (1)'],
                                   y=['Fully Paid (0)', 'Default (1)'],
                                   title=name)
                
                with cols[i]:
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
        with tab_metrics:
            st.header("Comprehensive Metrics Comparison")
            results_df = pd.DataFrame(st.session_state['model_results'])
            cols = ['Model', 'Accuracy', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall']
            results_df = results_df[cols]
            
            def highlight_max(s, props=''):
                return np.where(s == np.nanmax(s.values), props, '')
                
            st.dataframe(results_df.style.apply(highlight_max, props='background-color:#c6efce;', subset=['Accuracy', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall']))
            
            st.success("🎉 You have successfully completed the comparative study! You can now analyze these visuals to draw your final research conclusions.")
