# 🤖 Learning ML

A comprehensive hands-on machine learning journey featuring practical implementations of key algorithms, from supervised learning to clustering techniques and advanced ensemble methods.

## 📚 Contents

### 🔢 Regression Algorithms

#### [Linear_Regression.ipynb](Linear_Regression.ipynb)
Complete implementation of linear regression with real-world data (50 Startups dataset)
- Data preprocessing and feature encoding (OneHotEncoder)
- Train-test split and model training
- Multiple linear regression with categorical variables
- Model evaluation and predictions
- Visualization of results

#### [Ridge_&_Lasso_Reg.ipynb](Ridge_&_Lasso_Reg.ipynb)
Advanced regularization techniques for preventing overfitting (Boston Housing dataset)
- **Ridge Regression** - L2 regularization with alpha tuning
- **Lasso Regression** - L1 regularization for feature selection
- Automatic feature elimination using Lasso coefficients
- Comparative analysis: Linear vs Ridge vs Lasso
- Performance metrics: R² score, MSE, and MAE
- Feature importance and dimensionality reduction

### 🎯 Classification Algorithms

#### [Full_classification.ipynb](Full_classification.ipynb)
Comprehensive comparison of 6 major classification algorithms
- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Decision Tree Classifier** - Tree-based decision making
- **Random Forest Classifier** - Ensemble of decision trees
- **Naive Bayes** - Probabilistic classifier (BernoulliNB)
- **Support Vector Machine (SVM)** - Maximum margin classifier
- **Logistic Regression** - Linear classification model
- Feature scaling with StandardScaler
- Complete evaluation with accuracy scores and classification reports
- Model comparison and performance benchmarking

#### [TL_Classification_Code.ipynb](TL_Classification_Code.ipynb)
Practical classification workflow with real-world predictions (Telco Customer Churn dataset)
- End-to-end classification pipeline
- **KNN**, **Decision Tree**, and **Random Forest** implementations
- Feature scaling and data preprocessing
- Real-time predictions with new data
- Probability predictions using `predict_proba()`
- Model performance evaluation and accuracy metrics
- Hands-on approach to solving business problems

### 🌳 Ensemble Methods

#### [Bagging(Classifiers_and_Regressors).ipynb](Bagging(Classifiers_and_Regressors).ipynb)
In-depth exploration of Bootstrap Aggregating (Bagging) technique
- **Bagging Classifier** with Decision Trees and SVM base estimators
- **Bagging Regressor** for regression tasks
- Comparison: Bagging vs Random Forest
- **Pasting** (sampling without replacement) vs Bootstrap
- Hyperparameter tuning: `n_estimators`, `max_samples`, `bootstrap`
- Performance on synthetic datasets (make_classification, make_regression)
- Variance reduction and model stability
- Comprehensive metrics: Accuracy, R², MAE, MSE

#### [Bagging vs Random Forest.ipynb](Bagging%20vs%20Random%20Forest.ipynb)
Visual comparison and key differences between ensemble methods
- Side-by-side implementation of Bagging and Random Forest
- Decision tree visualization using `plot_tree()`
- Feature randomness in Random Forest vs Bagging
- Understanding `max_features` parameter
- Practical demonstration of algorithmic differences
- Visual analysis of decision boundaries

### 📊 Clustering Algorithms (Unsupervised Learning)

#### [Kmeans_Clustering.ipynb](Kmeans_Clustering.ipynb)
Customer segmentation using K-Means clustering (Mall Customer dataset)
- **Elbow Method** for optimal cluster selection (WCSS analysis)
- K-Means implementation with 6 clusters
- Customer segmentation by Annual Income and Spending Score
- Cluster visualization with centroids
- Market segmentation use case
- Color-coded scatter plots for cluster identification

#### [HC_Clustering.ipynb](HC_Clustering.ipynb)
Hierarchical clustering with dendrogram visualization
- **Dendrogram** creation using scipy's hierarchy module
- **Agglomerative Clustering** with Ward linkage
- Euclidean distance metric
- Bottom-up hierarchical approach
- Visual cluster analysis
- Customer grouping based on purchasing behavior

#### [Mean_shift_Clustering.ipynb](Mean_shift_Clustering.ipynb)
Non-parametric clustering algorithm exploration
- **Mean Shift** clustering with bandwidth parameter
- Automatic cluster number detection
- Density-based clustering approach
- Customer segmentation visualization
- Comparison with centroid-based methods
- Real-world application on mall customer data

### 🔧 Feature Engineering

#### [Feature_Eng_RFE.ipynb](Feature_Eng_RFE.ipynb)
Feature selection using Recursive Feature Elimination (Telco Customer Churn)
- **Recursive Feature Elimination (RFE)** with Logistic Regression
- Automated feature selection to improve model performance
- Comparing models with different feature counts (5 vs 14 features)
- Feature importance ranking
- Data preprocessing: handling missing values, one-hot encoding
- Model performance with full features vs selected features
- Reducing model complexity while maintaining accuracy
- Practical demonstration of the curse of dimensionality

### 📈 Time Series Forecasting & Analysis

#### [Time Series & Analysis/ARIMA.ipynb](Time%20Series%20&%20Analysis/ARIMA.ipynb)
Complete ARIMA modeling workflow for time series forecasting (Air Passengers dataset)
- **Stationarity Testing** - ADF (Augmented Dickey-Fuller) test
- **Time Series Decomposition** - Trend, Seasonal, and Residual components
- **Differencing** - Making time series stationary
- **ACF & PACF Plots** - Identifying AR and MA orders
- **ARIMA Model Building** - AutoRegressive Integrated Moving Average
- **Model Diagnostics** - Residual analysis and validation
- **Forecasting** - Predicting future values
- **Model Evaluation** - Performance metrics and visualization

#### [Time Series & Analysis/Decomposition.ipynb](Time%20Series%20&%20Analysis/Decomposition.ipynb)
Time series decomposition techniques and visualization
- **Additive Decomposition** - Breaking down time series into components
- **Seasonal Decomposition** - Extracting seasonal patterns
- **Trend Analysis** - Identifying long-term patterns
- Visualizing decomposed components
- Understanding seasonality and cyclical patterns

#### [Time Series & Analysis/TS_Facebook_Prophet.ipynb](Time%20Series%20&%20Analysis/TS_Facebook_Prophet.ipynb)
Facebook Prophet for time series forecasting
- **Prophet Model** - Facebook's forecasting tool
- **Automatic Seasonality Detection** - Daily, weekly, yearly patterns
- **Trend Changepoint Detection** - Identifying structural breaks
- **Forecasting with Uncertainty** - Confidence intervals
- **Component Visualization** - Trend and seasonal components
- **Future Predictions** - Making forecasts with Prophet
- Easy-to-use interface for business forecasting

#### [Time Series & Analysis/TS_Facebook_Prophet_Holidays.ipynb](Time%20Series%20&%20Analysis/TS_Facebook_Prophet_Holidays.ipynb)
Advanced Prophet modeling with holiday effects
- **Holiday Modeling** - Incorporating special events
- **Custom Seasonality** - Adding domain-specific patterns
- **Holiday Impact Analysis** - Measuring event effects
- Enhanced forecasting with business calendars

#### Projects
- **Project1 - Energy Forecasting** - Real-world energy consumption prediction
- **Project2 - Stock Market Forecasting** - Financial time series with Prophet
- **Project3 - Demand Forecasting for E-commerce** - Business demand prediction
- **UNI & Multi Variate Analysis** - Multi-variable forecasting with Prophet

## 🎯 Learning Objectives

This repository demonstrates practical mastery of:

### Supervised Learning
- **Regression**: Linear models, regularization techniques (Ridge/Lasso)
- **Classification**: 6+ algorithms from KNN to SVM, ensemble methods

### Unsupervised Learning
- **Clustering**: K-Means, Hierarchical, Mean Shift algorithms
- Customer segmentation and pattern discovery

### Time Series Analysis
- **Classical Methods**: ARIMA modeling, decomposition, stationarity testing
- **Modern Forecasting**: Facebook Prophet with seasonality and holidays
- **Real-World Projects**: Energy, stock market, and e-commerce forecasting
- ACF/PACF analysis and model diagnostics

### Advanced Techniques
- **Ensemble Methods**: Bagging, Random Forests, variance reduction
- **Feature Engineering**: RFE, dimensionality reduction, feature selection
- **Model Evaluation**: Cross-validation, multiple metrics, comparative analysis

### Best Practices
- Data preprocessing and feature scaling
- Train-test split methodology
- Hyperparameter tuning
- Model comparison and selection
- Visualization of results

## 🛠️ Tech Stack

**Core Libraries:**
- `scikit-learn` - Machine learning algorithms and utilities
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` & `seaborn` - Data visualization
- `scipy` - Scientific computing (hierarchical clustering)
- `statsmodels` - Time series analysis and statistical modeling
- `prophet` (fbprophet) - Facebook's forecasting library

**Key Modules Used:**
- `sklearn.linear_model` - Regression models
- `sklearn.ensemble` - Bagging, Random Forest
- `sklearn.tree` - Decision Trees
- `sklearn.cluster` - Clustering algorithms
- `sklearn.feature_selection` - RFE
- `sklearn.preprocessing` - StandardScaler, OneHotEncoder
- `sklearn.metrics` - Model evaluation
- `statsmodels.tsa` - ARIMA, decomposition, stationarity tests
- `fbprophet` - Prophet forecasting models

## 📊 Datasets Used

- **50 Startups** - Multi-variable linear regression
- **Boston Housing** - Regularization and feature selection
- **Telco Customer Churn** - Classification and RFE
- **Mall Customers** - Clustering and segmentation
- **Air Passengers** - Classic time series dataset for ARIMA
- **Custom Time Series** - Prophet forecasting demonstrations
- **Synthetic Data** - make_classification, make_regression

## 🚀 Getting Started

Each notebook is self-contained with:
1. **Data Loading** - Import and explore datasets
2. **Preprocessing** - Clean, encode, and scale features
3. **Model Training** - Implement algorithms with explanations
4. **Evaluation** - Metrics, visualizations, and insights
5. **Comparison** - Multiple approaches when applicable

## 📈 Key Insights

- **Regularization** reduces overfitting and can eliminate irrelevant features
- **Ensemble methods** generally outperform individual models
- **Feature selection** improves model interpretability and performance
- **Proper scaling** is crucial for distance-based algorithms
- **Multiple metrics** provide comprehensive model evaluation

## 💡 Use Cases Demonstrated

- 📈 Startup profit prediction
- 🏠 Housing price estimation
- 📞 Customer churn prediction
- 🛒 Customer market segmentation
- 🎯 Feature importance analysis
- ⚡ Energy demand forecasting
- 📊 Stock market prediction
- 🛍️ E-commerce demand planning
- 📅 Seasonal pattern detection

---

*A practical, code-first approach to understanding machine learning fundamentals through real implementations.*
