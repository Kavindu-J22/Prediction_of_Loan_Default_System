**Determinants and Prediction of Loan Default: A Comparative Study of
Logistic Regression and Machine Learning Models\
\
Introduction**

**1. Research Background**

In the modern financial landscape, retail banking institutions are
facing significant challenges due to rising credit risk exposure. This
is driven by economic volatility, the rapid expansion of digital lending
platforms, and increasing loan default rates. Accurate credit scoring is
essential for maintaining financial stability and making informed
lending decisions.

**2. The Current State of Credit Scoring**

Traditionally, financial institutions have relied on econometric models
like Logistic Regression for credit scoring. These models are favored
because of their high interpretability, transparency, and regulatory
acceptance. However, as loan portfolios become high-dimensional and
borrower behavior becomes more non-linear, these traditional models
often struggle to capture complex relationships within large and
unstructured datasets.

**3. The Role of Machine Learning**

Recent advancements in Artificial Intelligence have introduced Machine
Learning (ML) models---such as Random Forests, XGBoost, and
LightGBM---which offer significant advantages in predictive power and
scalability. While these models are known to outperform traditional
statistical methods in terms of accuracy and F1-score, they are often
viewed as \"black boxes\" due to their complexity.

**4. Problem Statement & Objectives**

There is a critical need to systematically evaluate whether the
transition from traditional econometric methods to advanced ML
algorithms provides statistically and economically meaningful
improvements. This study aims to bridge the existing research gap by
comparing a benchmark Logistic Regression model against ensemble ML
models using real-world retail banking datasets. By identifying the key
factors affecting loan default and evaluating model trade-offs, this
research provides insights into balancing interpretability with
predictive accuracy in credit risk management. **\
\
Steps to Execute Your Credit Risk Research**

**1. Data Preprocessing (Cleaning & Shaping)**

Before feeding data into the models, you need to ensure it is clean and
standardized:

- **Handling Missing Values**: Identify and fill or remove null entries
  in both numerical and categorical attributes.

- **Outlier Detection**: Use the **Interquartile Range (IQR)** method to
  detect and treat extreme values that could skew your results.

- **Categorical Encoding**: Convert text-based data (like loan grade or
  home ownership) into numerical representations that the machine can
  understand.

- **Feature Scaling**: Scale all numerical attributes to a standard
  range to ensure no single variable dominates the model due to its
  magnitude.

**2. Feature Selection (Identifying Key Drivers)**

Not all data points are equally important. You need to find the most
relevant attributes:

- **Correlation Analysis**: Measure the linear relationships between
  numerical features and the target variable (Loan Default).

- **Chi-square Tests**: Use these for categorical variables to assess
  their statistical dependence on the loan status.

**3. Model Development & Benchmarking**

This is the core of your comparative study:

- **The Baseline**: Start with **Logistic Regression** as your benchmark
  econometric model.

- **Advanced ML Models**: Develop ensemble models like **Random
  Forest**, **XGBoost**, and **LightGBM** to capture non-linear borrower
  behaviors.

- **Consistent Framework**: Ensure all models use the same preprocessing
  and feature engineering for a fair comparison.

**4. Rigorous Evaluation**

Compare the performance using multiple statistical metrics to see if ML
truly provides \"meaningful improvements\":

- **Core Metrics**: Evaluate based on **Accuracy**, **Precision**,
  **Recall**, and the **F1-Score**.

- **Predictive Power**: Use the **AUC-ROC** (Area Under the Receiver
  Operating Characteristic Curve) to measure how well the models
  distinguish between \"defaulters\" and \"non-defaulters\".

need to demostrate and visualize all steps and Option\'s in steamlit
dashbord also. (Dataset here in dataset folder)