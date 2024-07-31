#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor


# In[2]:


wind= pd.read_csv("wind energy updated.csv")


# In[3]:


wind


# In[4]:


wind["Datetime"]=pd.to_datetime(wind["Datetime"], utc=True)
wind["Datetime"]
wind.dtypes


# In[5]:


wind.info()


# In[6]:


wind['Datetime_update'] = wind['Datetime'].fillna(method='bfill')
wind


# In[7]:


wind["Datetime_update"].dt.year


# In[8]:


wind["Datetime_update"].dt.month


# In[9]:


wind["Datetime_update"].dt.day


# In[10]:


wind["year"]=wind["Datetime_update"].dt.year
wind["month"]=wind["Datetime_update"].dt.month
wind["day"]=wind["Datetime_update"].dt.day
wind


# In[11]:


wind["Datetime_update"].dt.hour


# In[12]:


wind["hour"]=wind["Datetime_update"].dt.hour
wind


# In[13]:


wind['UTC_Offset'] = wind['Datetime_update'].apply(lambda x: x.strftime('%z') if pd.notnull(x) else None)
wind


# In[14]:


wind.isnull().sum()


# In[15]:


print("Number of Unique values in offshore/onshore: ", wind['Offshore/onshore'].nunique())
print("Number of Unique values in Region: ", wind['Region'].nunique())
print("Number of Unique values in Grid connection type: ", wind['Grid connection type'].nunique())


# In[16]:


numerical = ['Measured & Upscaled', 'Day Ahead 11AM forecast', 'Day Ahead 11AM P10', 'Day Ahead 11AM P90', 'Week-ahead forecast', 'Week-ahead P10', 'Week-ahead P90', 'Load factor']
for col in numerical:
    wind[col].fillna(wind[col].mean(), inplace=True)
wind.isnull().sum()


# In[17]:


wind=wind.drop(['Datetime', 'Datetime_update', 'Resolution code', 'Decremental bid Indicator', 'UTC_Offset'], axis=1)
wind


# In[ ]:





# In[18]:


wind.replace({'Offshore': 0, 'Onshore': 1}, inplace=True)
wind.replace({'Flanders': 0, 'Federal': 1, 'Wallonia': 2}, inplace=True)
wind.replace({'Elia': 0, 'Dso': 1}, inplace=True)


# In[19]:


wind


# In[20]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix
corr_matrix = wind.corr()

# Create a heatmap
plt.figure(figsize=(18,20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=5.0)

# Display the heatmap
plt.title('Heatmap of Wind dataset with the Correlation matrix')
plt.show()


# In[21]:


X=wind.drop(['Most recent forecast', 'Most recent P10', 'Most recent P90', 'Day Ahead 11AM forecast', 'Day Ahead 11AM P90', 'Day-ahead 6PM P10', 'Day-ahead 6PM P90', 'Week-ahead P90'], axis=1)
X


# In[22]:


X = X.drop(range(1038575))
X


# In[23]:


y = X['Monitored capacity']
X = X.drop('Monitored capacity', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Standardization

# In[ ]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Support Vector Regressor

# In[24]:


# Support vector Regression

# Train an SVR model
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# R2 score
r2 = r2_score(y_test, y_pred)
#Mean Absolute error
mae = mean_absolute_error(y_test, y_pred)
# Root Mean square error
rmse = np.sqrt(mse)

# Scores 
print("R2 Score:", r2*100)
print("MAE:", mae)
print("Root Mean Square Error: ", rmse)


# In[25]:


# Initialize the SVR model
svr = SVR(kernel='rbf')

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svr, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

# Calculate RMSE for cross-validation
rmse_scores = np.sqrt(-scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(svr, X_train, y_train, cv=cv)

# Evaluate the model on cross-validated predictions
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

# Scores
print("Cross-Validated R2 Score:", r2*100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error: ", rmse)


# In[131]:


# Train an SVR model with 'rbf' kernel
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("SVR with Best Parameters")
print("R2 Score:", r2 * 100)
print("MAE:", mae)
print("RMSE:", rmse)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svr, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

# Calculate RMSE for cross-validation
rmse_scores = np.sqrt(-scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(svr, X_train, y_train, cv=cv)
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

print("Cross-Validated R2 Score:", r2 * 100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)

# Plot the actual vs predicted values
plt.figure(figsize=(12, 5))

# Plot for SVR test predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted (Test Set)')

# Plot for SVR cross-validated predictions
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted (Training Set - Cross-Validated)')

plt.tight_layout()
plt.show()


# In[26]:


svr = SVR(C=100, epsilon=0.5, gamma=0.01)

# Fit the model to the training data
svr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("SVR Model")
print("Best Parameters: {'C': 100, 'epsilon': 0.5, 'gamma': 0.01}")
print("R-squared Score:", r2 * 100)
print("MAE:", mae)
print("RMSE:", rmse)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(svr, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(svr, X_train, y_train, cv=cv)
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

print("Cross-Validated R2 Score:", r2 * 100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)

# Plot the actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted (Test Set)')

plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted (Training Set - Cross-Validated)')

plt.tight_layout()
plt.show()


# # Linear Regressor

# In[103]:


model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
#R2 score
r2 = r2_score(y_test, y_pred)
# Mean Absolute error
mae = mean_absolute_error(y_test, y_pred)
# Root Mean square error
rmse = np.sqrt(mse)

# Scores 
print("R2 Score:", r2*100)
print("MAE:", mae)
print("Root Mean Square Error: ", rmse)


# In[105]:


# Initialize the Linear Regression model
model = LinearRegression()

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Calculate cross-validation scores for MSE
mse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
# Calculate RMSE for cross-validation
rmse_scores = np.sqrt(-mse_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv)

# Evaluate the model on cross-validated predictions
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

# Scores
print("Cross-Validated R2 Score:", r2*100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)


# In[166]:


# Plot Actual vs Predicted for Test Data
plt.figure(figsize=(14, 6))

# Scatter plot for test data predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Test Data)')

# Scatter plot for cross-validated predictions
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Cross-Validated)')

plt.tight_layout()
plt.show()


# # Ada boost regressor

# In[106]:


base_estimator = DecisionTreeRegressor(max_depth=4)
adaboost = AdaBoostRegressor(base_estimator=base_estimator, 
                             n_estimators=50, 
                             learning_rate=1.0, 
                             random_state=42)
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
#R2 score
r2 = r2_score(y_test, y_pred)
#Mean absolute error
mae = mean_absolute_error(y_test, y_pred)
# Root Mean square error
rmse = np.sqrt(mse)



print("R2:", r2*100)
print("MAE:", mae)
print("Root Mean Square Error: ", rmse)


# In[110]:


# Initialize the base estimator and AdaBoost model

# Define the base estimator with the desired hyperparameters
base_estimator = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, min_samples_split=10)

# Initialize the AdaBoostRegressor with the base estimator and other parameters
adaboost = AdaBoostRegressor(base_estimator=base_estimator, learning_rate=0.01, n_estimators=200)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate cross-validation scores for MSE
mse_scores = cross_val_score(adaboost, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
# Calculate RMSE for cross-validation
rmse_scores = np.sqrt(-mse_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(adaboost, X_train, y_train, cv=cv)

# Evaluate the model on cross-validated predictions
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

# Scores
print("Cross-Validated R2 Score:", r2*100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)


# In[111]:


# Plot Actual vs Predicted for Test Data
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Data: Actual vs Predicted')

# Plot Actual vs Cross-Validated Predictions for Training Data
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, alpha=0.7, edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Cross-Validated Predictions')
plt.title('Training Data: Actual vs Cross-Validated Predictions')

plt.tight_layout()
plt.show()


# # Ridge Regressor

# In[112]:


# Create and train the Ridge Regression model
alpha = 1.0  # Regularization strength
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)

# Make predictions
y_pred = ridge.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
#R2 Score
r2 = r2_score(y_test, y_pred)
# Mean Absolute error
mae = mean_absolute_error(y_test, y_pred)
# Root Mean square error
rmse = np.sqrt(mse)

print("R-squared Score:", r2*100)
print("MAE:", mae)
print("RMSE: ", rmse)


# In[113]:


# Initialize the Ridge Regression model
alpha = 1.0  # Regularization strength
ridge = Ridge(alpha=alpha)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate cross-validation scores for MSE
mse_scores = cross_val_score(ridge, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
# Calculate RMSE for cross-validation
rmse_scores = np.sqrt(-mse_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(ridge, X_train, y_train, cv=cv)

# Evaluate the model on cross-validated predictions
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

# Scores
print("Cross-Validated R2 Score:", r2*100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)


# In[161]:


# Plot the actual vs predicted values
plt.figure(figsize=(12, 5))

# Plot for Ridge Regression test predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression: Actual vs Predicted (Test Set)')

# Plot for Ridge Regression cross-validated predictions
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression: Actual vs Predicted (Training Set - Cross-Validated)')

plt.tight_layout()
plt.show()


# # RANSAC

# In[158]:


# Create and train the RANSAC Regression model
ransac = RANSACRegressor()
ransac.fit(X_train, y_train)

# Make predictions
y_pred = ransac.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RANSAC Regression")
print("R-squared Score:", r2 * 100)
print("MAE:", mae)
print("RMSE:", rmse)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(ransac, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(ransac, X_train, y_train, cv=cv)
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

print("Cross-Validated R2 Score:", r2 * 100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)

# Plot the actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('RANSAC Regression: Actual vs Predicted (Test Set)')

plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('RANSAC Regression: Actual vs Predicted (Training Set - Cross-Validated)')

plt.tight_layout()
plt.show()


# # LASSO regression

# In[157]:


# Create and train the Lasso Regression model
alpha = 1.0  # Regularization strength
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Lasso Regression")
print("R-squared Score:", r2 * 100)
print("MAE:", mae)
print("RMSE:", rmse)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(lasso, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", rmse_scores.mean())

# Make cross-validated predictions
y_cv_pred = cross_val_predict(lasso, X_train, y_train, cv=cv)
mse = mean_squared_error(y_train, y_cv_pred)
r2 = r2_score(y_train, y_cv_pred)
mae = mean_absolute_error(y_train, y_cv_pred)
rmse = np.sqrt(mse)

print("Cross-Validated R2 Score:", r2 * 100)
print("Cross-Validated MAE:", mae)
print("Cross-Validated Root Mean Square Error:", rmse)

# Plot the actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted (Test Set)')

plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted (Training Set - Cross-Validated)')

plt.tight_layout()
plt.show()


# # Huber regression

# In[ ]:


# Train the Huber Regressor model
huber = HuberRegressor()
huber.fit(X_train_scaled, y_train)

# Make predictions
y_pred_huber = huber.predict(X_test_scaled)

# Evaluate the model
mse_huber = mean_squared_error(y_test, y_pred_huber)
r2_huber = r2_score(y_test, y_pred_huber)
mae_huber = mean_absolute_error(y_test, y_pred_huber)
rmse_huber = np.sqrt(mse_huber)

print("Huber Regressor R2:", r2_huber * 100)
print("Huber Regressor MAE:", mae_huber)
print("Huber Regressor Root Mean Square Error:", rmse_huber)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred_huber = cross_val_predict(huber, X_train_scaled, y_train, cv=cv)

# Plot the actual vs predicted values
plt.figure(figsize=(12, 5))

# Plot for test set
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_huber, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Huber Regressor: Actual vs Predicted (Test Set)')

# Plot for training set (cross-validated)
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_cv_pred_huber, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Huber Regressor: Actual vs Predicted (Training Set - Cross-Validated)')

plt.tight_layout()
plt.show()


# In[ ]:




