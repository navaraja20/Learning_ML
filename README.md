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

#### [Feature_Eng_SFS.ipynb](Feature_Eng_SFS.ipynb) & [mlXtend_Feature_Eng_SFS.ipynb](mlXtend_Feature_Eng_SFS.ipynb)
Sequential Feature Selection for optimal feature subsets
- **Sequential Forward Selection (SFS)** - Adding features iteratively
- **Sequential Backward Selection** - Removing features iteratively
- Using mlXtend library for advanced feature selection
- Comparing model performance across different feature combinations
- Practical application on Telco Customer Churn dataset
- Greedy search algorithms for feature optimization

#### [FE_PCA.ipynb](FE_PCA.ipynb)
Principal Component Analysis for dimensionality reduction
- **PCA (Principal Component Analysis)** - Linear dimensionality reduction
- Variance explanation and component selection
- Reducing feature space while retaining information
- Data standardization before PCA
- Visualization of principal components
- Model performance comparison: full features vs PCA-reduced features
- Handling high-dimensional datasets efficiently

#### [FE_LDA.ipynb](FE_LDA.ipynb)
Linear Discriminant Analysis for supervised dimensionality reduction
- **LDA (Linear Discriminant Analysis)** - Supervised feature extraction
- Maximizing class separability
- Comparison with PCA (unsupervised vs supervised)
- Feature projection for classification tasks
- Multi-class discrimination optimization
- Enhanced classification performance through LDA

#### [FE_KPCA_&_QDA.ipynb](FE_KPCA_&_QDA.ipynb)
Advanced dimensionality reduction and classification techniques
- **Kernel PCA (KPCA)** - Non-linear dimensionality reduction
- RBF, polynomial, and sigmoid kernels
- **Quadratic Discriminant Analysis (QDA)** - Non-linear classifier
- Handling non-linearly separable data
- Kernel trick for complex feature spaces
- Performance comparison: LDA vs QDA

#### [chi_square.ipynb](chi_square.ipynb)
Statistical feature selection using Chi-Square test
- **Chi-Square Test** - Statistical feature selection for categorical data
- SelectKBest for automated feature ranking
- P-value analysis for feature importance
- Statistical significance testing
- Feature selection for classification tasks
- Identifying most relevant features statistically

### ⚙️ Hyperparameter Optimization

#### [HPO_BreastCancer.ipynb](HPO_BreastCancer.ipynb)
Comprehensive hyperparameter tuning techniques (Breast Cancer dataset)
- **Grid Search CV** - Exhaustive hyperparameter search
- **Random Search CV** - Randomized parameter sampling
- **Bayesian Optimization** - Smart hyperparameter tuning
- Cross-validation strategies
- Random Forest hyperparameter tuning
- Finding optimal model configurations
- Performance metrics and model comparison
- Avoiding overfitting through proper validation

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

## 🧠 Deep Learning

### Convolutional Neural Networks (CNNs)

#### [Deep Learning/CNN_Image_Classification.ipynb](Deep%20Learning/CNN_Image_Classification.ipynb)
Building CNNs from scratch for image classification (Fashion MNIST)
- **CNN Architecture** - Conv2D, MaxPooling2D, Dropout layers
- Fashion MNIST dataset (10 clothing categories)
- Image preprocessing and normalization
- Model compilation with categorical crossentropy
- Training deep learning models with TensorFlow/Keras
- Evaluation metrics and accuracy visualization
- Understanding convolutional filters and feature maps

#### [Deep Learning/Image_Augmentation.ipynb](Deep%20Learning/Image_Augmentation.ipynb)
Data augmentation techniques to improve model generalization
- **Image Augmentation** - Rotation, flipping, zooming, shifting
- Preventing overfitting through data diversity
- Real-time data augmentation during training
- Keras ImageDataGenerator utilities
- Visualizing augmented images
- Expanding limited datasets artificially

#### [Deep Learning/Pre_trained_Models.ipynb](Deep%20Learning/Pre_trained_Models.ipynb)
Transfer learning with pre-trained models
- **Pre-trained Models** - VGG16, ResNet, MobileNet
- ImageNet weights for transfer learning
- Feature extraction from pre-trained networks
- Fine-tuning strategies
- Quick deployment with minimal training
- Leveraging models trained on millions of images

#### [Deep Learning/X_ray_Image_Classification_with_PreTrained_Models.ipynb](Deep%20Learning/X_ray_Image_Classification_with_PreTrained_Models.ipynb)
Medical image classification using transfer learning (Chest X-ray dataset)
- **Medical Image Classification** - Pneumonia detection from X-rays
- Transfer learning with VGG16/ResNet for medical imaging
- Handling imbalanced medical datasets
- Data augmentation for medical images
- Model evaluation with precision, recall, F1-score
- Real-world healthcare application
- Binary classification: Normal vs Pneumonia
- Deployment-ready medical AI model

### Recurrent Neural Networks (RNNs)

#### [Deep Learning/TATA_stock_prediction_LSTM.ipynb](Deep%20Learning/TATA_stock_prediction_LSTM.ipynb)
Stock price prediction using LSTM networks
- **LSTM (Long Short-Term Memory)** - Sequential data modeling
- Time series prediction with deep learning
- Stock market price forecasting (TATA Global)
- Sequence creation with sliding windows
- MinMax scaling for neural networks
- Multi-layer LSTM architecture
- Dropout for regularization
- Predicting future stock prices
- Visualization of predictions vs actual prices

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
- **Feature Engineering**: RFE, SFS, PCA, LDA, KPCA, Chi-Square
- **Hyperparameter Optimization**: Grid Search, Random Search, Bayesian Optimization
- **Model Evaluation**: Cross-validation, multiple metrics, comparative analysis

### Deep Learning
- **CNNs**: Image classification, convolutional architectures, transfer learning
- **RNNs/LSTMs**: Sequential data, time series prediction, stock forecasting
- **Transfer Learning**: Pre-trained models, fine-tuning, medical imaging
- **Data Augmentation**: Image preprocessing, generalization techniques

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
- `tensorflow` & `keras` - Deep learning frameworks
- `mlxtend` - Machine learning extensions for advanced feature selection

## 📊 Datasets Used

- **50 Startups** - Multi-variable linear regression
- **Boston Housing** - Regularization and feature selection
- **Telco Customer Churn** - Classification, RFE, and feature engineering
- **Mall Customers** - Clustering and segmentation
- **Breast Cancer Wisconsin** - Hyperparameter optimization
- **Air Passengers** - Classic time series dataset for ARIMA
- **Custom Time Series** - Prophet forecasting demonstrations
- **Fashion MNIST** - CNN image classification (60,000 images)
- **Chest X-ray Images** - Medical image classification (Pneumonia detection)
- **TATA Stock Data** - LSTM stock price prediction
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
- 🏥 Medical diagnosis (Pneumonia from X-rays)
- 👕 Fashion item classification
- 💹 Stock price forecasting with deep learning
- 🔬 Cancer detection and classification

---

*A comprehensive, hands-on journey through machine learning - from classical algorithms to cutting-edge deep learning, covering regression, classification, clustering, time series analysis, and neural networks.*
