# Study - Machine Learning Regression

## Medical Cost Personal Datasets
<div>
  Dataset link : https://www.kaggle.com/datasets/mirichoi0218/insurance
</div

<div>
  Is it possible to predict insurance costs? Let's mine this data, extract and analyze the results.
</div>
<div>
  In Machine Learning, we have Regression Algorithms used to predict continuous target values. As the name implies, we can use statistical regression techniques, such as linear and multiple regression. In these cases, we use the values from the database to extract patterns and, through analyses and statistical tests, draw a line (or curve) that describes the distribution of this data, using its function to predict future values.
</div>

## Columns
<div>
age: age of primary beneficiary


sex: insurance contractor gender, female, male

bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

children: Number of children covered by health insurance / Number of dependents

smoker: Smoking

region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

charges: Individual medical costs billed by health insurance
</div>

<div>
  In this case, target variable is "Charges". How the other variables will be treated depends on the algorithm of ML.
</div>

## Data Exploration and Preprocessing

<div>
  An extremely important step for any Machine Learning algorithm construction. Using pandas resources, we verify the existence of null values, statistical measures and possible outliers.
</div>

<div>
  To analyze the BoxPlot, I used Plotly library. The search for outlier values can be analyzed through the separation of the quartiles.
</div>

### Linear Correlation

<div>
Linear correlation is a statistical measure that indicates the degree of relationship between two quantitative variables. The correlation can be positive, negative, or nonexistent. This relationship can be representated by linear equation y = mx + b. It calculates the value and direction of the linear relationship between two quantitative variables. The coefficients range from -1 to 1, where -1 indicates a perfect negative correlation, 1 indicates a perfect positive correlation, and 0 indicates no correlation.
</div>
<div>
  Procedure for calculating and analyzing the linear correlation between two variables:

  - Exploratory analysis of the scatter plot between the two variables (using graphic resources from MatPlotLib
  - Calculation of the linear coefficient (method depends of the kind of distribuition)
  - Perform the hypothesis test for correlation
</div>
<div>
  In this case, I also analyzed the normality of the distribution of values and also the histogram. We can see, graphically, if the distribution approaches normal or not. We can also confirm through hypothesis tests whether the distribution is truly normal or not. 
  
  A normal Q-Q plot is a graphical tool used to assess whether a dataset follows a normal distribution. It compares the quantiles (or percentiles) of the observed data with the expected quantiles of a theoretical normal distribution. Points that closely align with a straight line indicate that the data is approximately normal, while deviations from this line suggest non-normality. "scipy.stats" provides a wide range of distributions (e.g., Normal, Exponential, Binomial) with methods to work with them and shows it in graphics; Plotly is used to shows graphics about the hystogram distributions. Tests used to verify normal distribuition: Shapiro-Wilk and Lilliefors Test (statsmodels library)
</div>

### SHAPIRO- WILK TEST 
<div>
  
The Shapiro-Wilk test is a statistical tool whose purpose is to check if a sample of data follows a normal distribution.)

- H0 (null hypothesis) : Normal Distribution : p>0.05

- Ha (alternative hypothesis) : Non-Normal Distribution : p<=0.05

- Limit = only serves up to 5000 records (lines)
</div>

### LILLIEFORS TEST (kolmogorov_ Sminorv)

<div>
The Lilliefors test is a normality test that is an improvement of the Kolmogorov-Smirnov (K-S) test. It is used to test if data come from a normally distributed population (when the population mean and variance are not specified).

- H0 (null hypothesis) : Normal Distribution : p>0.05

- Ha (alternative hypothesis) : Non-Normal Distribution : p<=0.05
</div>

### Linear Correlation tests

<div>
TESTS:

- Pearson (normal distribution)
- Spearman (non-normal distribution)
- Kendall (non-normal distribution with a small number of samples)

In this case, we cannot use Pearson's Test because the distribution of the samples is not normal. I used Spearman Test.

- H0 (null hypothesis) :  there is no linear correlation: p>0.05
- Ha (alternative hypothesis) : there is a linear correlation: p<=0.05
  
</div>

## Machile Learning Algorithms used

### SIMPLE LINEAR REGRESSION
<div>
  It is a supervised learning algorithm that seeks to model the relationship between a dependent variable and one or more independent variables. In simple linear regression, there is only one independent variable, while in multiple regression, more than one independent variable can be used. It is represented by a line in a Cartesian plane (linear function y = mx + b) and is used to predict or forecast a dependent variable. The SkLearn library facilitates the implementation of linear regression.
  
  There are some metrics available to evaluate the performance of a regression algorithm, such as the coefficient of determination (R^2), mean squared error (MSE), and root mean squared error (RMSE).The coefficient of determination, referred to as R2, measures the degree to which a regression model explains the variability of the target variable.

  The R2 expresses the amount of variance in the data that is explained by the linear model. Thus, the higher the R², the more explanatory the linear model is, meaning it fits the sample better.

</div>
<div>
  
  #### After applying the algorithm , test and train model, we must evaluate our results:

 - Calculate the residuals (using statsmodels and OLS): a residual is the difference between the observed value of a dependent variable (Y) and the value predicted by the model for that variable, given a specific value of the independent variable(s).
 
 -  Calculete Normality tests for residuals: this step assess whether the error terms in a statistical model, specifically after fitting the model, follow a normal distribution. This is a crucial assumption for many statistical methods, including linear regression, where normally distributed residuals are essential for accurate hypothesis testing and confidence interval estimation.
   
 -  Analysis of the Homoscedasticity of the Residuals : The analysis of homoscedasticity in the residuals of a linear regression model assesses whether the variance of the errors (residuals) is constant for all values of the independent variables.
   
 -  Outliers of the residues  (it should be between 3 and -3)

   A consistent model must  has normally distributed residual, homoscedasticity in the residuals and residuals's outliers within the indicated range.

</div>

### MULTIPLE LINEAR REGRESSION

<div>
  In Multiple Linear Regression, we use more independent variables. We adjust the Regression Linear Algorithm and use the other methods to evaluate the results.
  
</div>
<div>
  In the evaluate's step, we must add the teste : Absence of Multicollinearity.

  The absence of multicollinearity is crucial for the validity of regression analysis. It ensures that the model accurately reflects the relationships between the independent and dependent variables. When multicollinearity is present, it can lead to inflated standard errors, incorrect sign interpretations, and unreliable coefficient estimates. 

  If R^2 is high and the partial correlations are low, multicollinearity is a possibility. When this happens, one or more variables may be unnecessary in the model. If R^2 is high and the partial correlations are also high, multicollinearity may not be immediately detectable.
</div>

### POLYNOMIAL REGRESSION

<div>
  Polynomial Regression is a type of regression. The relationship between the independent variable and the dependent variable is modeled as an polynomial function (n-th degree). 

  Simple Linear Regression is adapted to polynomial regression by raising the independent variable to the chosen degree.

</div>

### SUPPORT VECTOR REGRESSION

<div>
  Support Vector Regression (SVR) is a machine learning algorithm used for regression tasks. It aims to find a function that best predicts a continuous target variable. SVR maps input variables into a high-dimensional feature space and finds a hyperplane that maximizes the margin between itself and the closest data points while minimizing prediction error. This  is particularly useful for non-linear relationships between variables and can be adapted through the use of different kernel functions. In this case, we have a non-linear relationship, as we tested before.

  The SkLearn provides the model  to fit our variables of train and test. The hyperparameter used to configurate is "kernel".

#### StandardScaler

<div>
  The StandardScaler is a preprocessing tool in Python's scikit-learn library that standardizes features by removing the mean and scaling to unit variance. It’s commonly used in machine learning pipelines to ensure that features are on the same scale, which can improve the performance of many algorithms.

  I tested the results using StandardScaler and not using it , to compare the results.

</div>

### REGRESSION WITH XGBOOST

<div>
  XGBoost (Extreme Gradient Boosting) is a powerful and efficient implementation of the gradient boosting algorithm, widely used for regression predictive modeling. 

  To use XGBoost, you need to install the library. Line code : !pip install xgboost

  Hyperparameters used: n_estimators, max_depth, learning_rate, objective.
</div>

### REGRESSION WITH LIGHTGBM

<div>
  LightGBM (Light Gradient Boosting Machine), is a high-performance gradient boosting framework designed for efficient and scalable machine learning tasks. It supports regression, classification, and ranking tasks. For multivariate regression, LightGBM can handle multiple target variables by training separate models for each target or by using custom approaches to predict multiple outputs simultaneously.

  To use LightGBM, you need to install the library. Line code :!pip install lightgbm

  Hyperparameters used: num_leaves, max_depth, learning_rate, n_estimators.
</div>

###  REGRESSION WITH CATBOOST

<div>
  CatBoost is a high-performance, open-source gradient-boosting framework developed by Yandex, particularly effective for regression tasks. It excels in handling categorical features efficiently, making it suitable for various machine learning applications. The Catboost algorithm performs its own replacement of nominal variables.

  To use CatBoost, you need to install the library. Line code : !pip install catboost

  Hyperparameters used: iterations, learning_rate, depth, random_state
</div>

### ARTIFICIAL NEURAL NETWORKS: REGRESSION

<div>
  An artificial neural network (ANN) is a machine learning model inspired by the human brain's structure and function. It consists of interconnected nodes or neurons organized in layers: an input layer, one or more hidden layers, and an output layer. Each node has associated weights and thresholds. When the output of a node exceeds a certain threshold, it activates and passes data to the next layer.
  
  ANNs can be effectively used for regression tasks by modeling complex relationships between input variables and their corresponding output values. I used MLPRegressor from SkLearn.

  Hyperparameters used : hidden_layer_sizes, activation, verbose,max_iter,solver, random_state
</div>

### Metrics

<div>
  The algorithms based on linear regression, in addition to the metrics, are also evaluated by the statistical tools mentioned earlier (to analyze the model's consistency). The other regression algorithms use only the metrics R^2, Mean Squared Error (MSE), and also Root Mean Squared Error (RMSE).

</div>

#### Cross Validation

<div>
  Cross-validation is a statistical technique used to evaluate the performance of machine learning models, particularly to assess how well a model generalizes to unseen data. It involves partitioning the dataset into multiple subsets, training the model on some subsets, and evaluating it on the remaining subsets. This process is repeated multiple times, with different subsets used for training and evaluation, to provide a more robust estimate of the model's performance. 

  I used this CV after train and test results, to validate the previous results. Separated the data into folds, with the parameters : n_splits=15, shuffle = True, random_state=5

  
</div>

## RESULTS

<div>

- Simple Linear Regression: RMSE: 7026.60, R^2 improved by 61%, CROSS-VALIDATION: 60.33%

- Multiple Regression: RMSE 5771.6, R^2 73.06%, test: 79.11%, Cross-validation: 73.60%

- Polynomial Regression: RMSE: 7145.00, R^2 59.34%, TEST: 67.13%

- Support Vector Regression: RMSE: 4412.3 R^2: 84.20% and test 87.8% cross-validation: 83.50%

- Decision Tree Regression: RMSE: 4328.25, R^2: 85.3% and test: 88.3%, CROSS VALIDATION: 83.04%

- Random Forest Regression: RMSE: 4331.68, Cross-validation: 86%, R^2: 90.5%, test: 88.23%

- XGBoost Regression: RMSE: 4392.58, R^2 92.4% and test: 89%, Cross-validation: 84.4%

- LightGBM Regression: RMSE: 4093.6, R^2 89% and test: 88.5%, cross-validation: 85.21%

- CatBoost Regression: RMSE: 4093.34, R^2: 87.3% and test 89.3%

- Neural Networks : scaled variables, RMSE: 4618.8, R^2: 88.1% and test score: 86.6%

- BEST: XGBoost Regression: RMSE: 4392.58, R^2 92.4% and test: 89%, Cross-validation: 84.4
</div>
