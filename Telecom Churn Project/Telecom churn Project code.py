#!/usr/bin/env python
# coding: utf-8

# #                                         P383-Telecom-Churn

# # Problem Statement

# 1.Customer churn is a big problem for telecommunications companies. Indeed, their annual churn rates are usually higher than 10%. 
# 
# 2.For that reason, they develop strategies to keep as many clients as possible.
# 
# 3.This is a classification project since the variable to be predicted is binary (churn or loyal customer).
# 
# 4.The goal here is to model churn probability, conditioned on the customer features.

# In[1]:


# Import Libraries

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# # Load Dataset
# 

# In[2]:


Telecom_churn_data = pd.read_csv("Churn (1).csv")


# # Statistical Analysis

# In[3]:


Telecom_churn_data.head(5)


# In[4]:


# Rename columns 

new_column_names = {
    'area.code' : 'area_code',
    'account.length': 'account_length',
    'voice.plan': 'voice_plan',
     'voice.messages': 'voice_messages',
     'intl.plan': 'intl_plan',
     'intl.mins': 'intl_mins',
     'intl.calls': 'intl_calls',
     'intl.charge': 'intl_charge',
     'day.mins': 'day_mins',
     'day.calls': 'day_calls',
     'day.charge': 'day_charge',
     'eve.mins': 'eve_mins',
     'eve.calls': 'eve_calls',
     'eve.charge': 'eve_charge',
     'night.mins': 'night_mins',
     'night.calls': 'night_calls',
     'night.charge': 'night_charge',
     'customer.calls': 'customer_calls'
}


Telecom_churn_data = Telecom_churn_data.rename(columns= new_column_names)


# In[5]:


Telecom_churn_data.dtypes


# In[6]:


Telecom_churn_data.info()


# 1.when we observe "day_charge" and "eve_mins" columns. Both columns have numerical data but We observe in
#  Telecom_churn_data.info()  Data Types are object
#  
# 2.So, we need to convert them into Float Data types

# In[7]:



columns_to_convert = ['day_charge', 'eve_mins']

Telecom_churn_data[columns_to_convert] = Telecom_churn_data[columns_to_convert].apply(pd.to_numeric, errors='coerce')


# In[8]:


Telecom_churn_data.info()


# In[9]:


Telecom_churn_data.shape


# In[10]:


Telecom_churn_data.columns


# In[11]:


# Here 'Unnamed: 0' is index column so, drop 'Unnamed: 0' column 
Telecom_churn_data = Telecom_churn_data.drop(columns = 'Unnamed: 0')


# In[12]:


Telecom_churn_data.describe()


# In[13]:


Telecom_churn_data.head(2)


# In[14]:


Telecom_churn_data.isna().sum()


# We can observe day_charge and eve_mins columns we have null values

# In[15]:


# Impute null values with median
Telecom_churn_data.fillna(Telecom_churn_data.median(), inplace=True)


# In[16]:


Telecom_churn_data.isna().sum()


# In[17]:


# checking weather is having duplicates or not

Telecom_churn_data[Telecom_churn_data.duplicated()]


# In[18]:


# checking Unique Values
Telecom_churn_data['state'].unique()


# In[19]:


# checking Unique Values
Telecom_churn_data['area_code'].unique()


# In[20]:


# checking Unique Values
Telecom_churn_data['intl_plan'].unique()


# In[21]:


# checking Unique Values
Telecom_churn_data['intl_plan'].unique()


# In[22]:


Telecom_churn_data.head(2)


# In[23]:


# checking Unique Values
Telecom_churn_data['voice_plan'].unique()


# In[24]:


# checking Unique Values
Telecom_churn_data['customer_calls'].unique()


# In[25]:


# checking Unique Values
Telecom_churn_data['churn'].unique()


# In[26]:


# checking Unique Values
Telecom_churn_data['intl_plan'].unique()


# In[27]:


Telecom_churn_data.corr()


# # Exploratory Data Analysis

# In[28]:


plt.title('Distribution of the churn columns')
explode = (0, 0.1)
Telecom_churn_data['churn'].value_counts().plot(kind='pie', autopct = '%.2f%%',
            explode = explode, colors=['#3ec300', '#ff1d15'])


# In[29]:


plt.title('Distribution of the churn columns')
explode = (0, 0.1)
Telecom_churn_data['voice_plan'].value_counts().plot(kind='pie', autopct = '%.2f%%',
            explode = explode, colors=['#3ec300', '#ff1d15'])


# In[30]:


plt.title('Distribution of the churn columns')
explode = (0, 0.1)
Telecom_churn_data['intl_plan'].value_counts().plot(kind='pie', autopct = '%.2f%%',
            explode = explode, colors=['#3ec300', '#ff1d15'])


# In[31]:




top_5_countries = Telecom_churn_data['state'].value_counts().head(5).index
ch_top_5 = Telecom_churn_data[Telecom_churn_data['state'].isin(top_5_countries)]

# Create count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=ch_top_5, x='state', hue='churn')
plt.title('Top 5 Countries with Churn')
plt.xlabel('Country')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.xticks(rotation=45)
plt.show()


# In[32]:



# Select top 5 area codes based on churn count
top_5_area_codes = Telecom_churn_data['area_code'].value_counts().head(5).index

# Filter DataFrame for top 5 area codes
ch_top_5_area_codes = Telecom_churn_data[Telecom_churn_data['area_code'].isin(top_5_area_codes)]

# Create FacetGrid
g = sns.FacetGrid(ch_top_5_area_codes, row='area_code', col='state', hue='churn', margin_titles=True)
g.map(sns.countplot, 'churn', order=['yes', 'no'])

# Adjust plot aesthetics
g.set_axis_labels('Churn', 'Count')
g.set_titles(row_template='Area Code: {row_name}', col_template='State: {col_name}')
g.add_legend(title='Churn')
plt.tight_layout()
plt.show()


# Observation 
# 
# 1. In first Row we can Observe Count of churn in all the States with in area_code_415 pincode
# 
# 2. In Second Row we can Observe Count of churn in all the States with in area_code_408 pincode
# 
# 3. In Third Row we can Observe Count of churn in all the States with in area_code_510 pincode

# In[33]:



# Define the list of categorical columns
categorical_columns = ['state', 'area_code', 'voice_plan', 'intl_plan', 'churn']

# Assuming Telecom_churn_data is your DataFrame
Telecom_churn_data1 = Telecom_churn_data.copy()

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns to numerical values
for col in categorical_columns:
    Telecom_churn_data1[col] = label_encoder.fit_transform(Telecom_churn_data1[col])

# Display the modified DataFrame
print(Telecom_churn_data1)


# In[34]:


Telecom_churn_data1.head(2)


# In[35]:


sns.boxplot(data=Telecom_churn_data1, orient='h')


# In[36]:


# Create a figure and axes
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

# Plot the first box plot
sns.boxplot(x='voice_messages', data=Telecom_churn_data1, ax=axes[0])
axes[0].set_title('voice_messages')

# Plot the second box plot
sns.boxplot(x='voice_plan', data=Telecom_churn_data1, ax=axes[1])
axes[1].set_title('voice_plan')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[37]:


# Create a figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 10))


sns.boxplot(x='intl_mins', data=Telecom_churn_data1, ax=axes[0])
axes[0].set_title('international mins')


sns.boxplot(x='intl_calls', data=Telecom_churn_data1, ax=axes[1])
axes[1].set_title('International Calls')


sns.boxplot(x='intl_charge', data=Telecom_churn_data1, ax=axes[2])
axes[2].set_title('internationalcharges')
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[38]:


# Create a figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 10))


sns.boxplot(x='eve_mins', data=Telecom_churn_data1, ax=axes[0])
axes[0].set_title('Evening minutes')


sns.boxplot(x='eve_calls', data=Telecom_churn_data1, ax=axes[1])
axes[1].set_title('Evening Calls')


sns.boxplot(x='eve_charge', data=Telecom_churn_data1, ax=axes[2])
axes[2].set_title('Evening charges')
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[39]:


# Create a figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 10))


sns.boxplot(x='night_mins', data=Telecom_churn_data1, ax=axes[0])
axes[0].set_title('Night mins')


sns.boxplot(x='night_calls', data=Telecom_churn_data1, ax=axes[1])
axes[1].set_title('Night Calls')


sns.boxplot(x='night_charge', data=Telecom_churn_data1, ax=axes[2])
axes[2].set_title('Night charges')
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# # Checking Weather Data is following Normal Distribution or not

# In[40]:



# Create a figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# Plot distribution plot for 'intl_mins'
sns.histplot(Telecom_churn_data1['intl_mins'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of International Minutes')

# Plot distribution plot for 'intl_calls'
sns.histplot(Telecom_churn_data1['intl_calls'], kde=True, ax=axes[1])
axes[1].set_title('Distribution of International Calls')

# Plot distribution plot for 'intl_charge'
sns.histplot(Telecom_churn_data1['intl_charge'], kde=True, ax=axes[2])
axes[2].set_title('Distribution of International Charges')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[41]:



# Create a figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# Plot distribution plot for 'day_mins'
sns.histplot(Telecom_churn_data1['day_mins'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Day Minutes')

# Plot distribution plot for 'day_calls'
sns.histplot(Telecom_churn_data1['day_calls'], kde=True, ax=axes[1])
axes[1].set_title('Distribution of Day Calls')

# Plot distribution plot for 'day_charge'
sns.histplot(Telecom_churn_data1['day_charge'], kde=True, ax=axes[2])
axes[2].set_title('Distribution of Day Charges')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[42]:



# Create a figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# Plot distribution plot for 'eve_mins'
sns.histplot(Telecom_churn_data1['eve_mins'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Evening Minutes')

# Plot distribution plot for 'eve_calls'
sns.histplot(Telecom_churn_data1['eve_calls'], kde=True, ax=axes[1])
axes[1].set_title('Distribution of  Evening Calls')

# Plot distribution plot for 'eve_charge'
sns.histplot(Telecom_churn_data1['eve_charge'], kde=True, ax=axes[2])
axes[2].set_title('Distribution of  Evening Charges')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[43]:



# Create a figure and axes
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Plot distribution plot for 'account_length'
sns.histplot(Telecom_churn_data1['account_length'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Account Length')

# Plot distribution plot for 'Voice_mossages'
sns.histplot(Telecom_churn_data1['voice_messages'], kde=True, ax=axes[1])
axes[1].set_title('Distribution of  Voice Messages')



# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[44]:



plt.figure(figsize=(10, 6))

# Plot histograms
sns.histplot(Telecom_churn_data1['day_mins'], color='blue', label='Day Minutes', kde=True)
sns.histplot(Telecom_churn_data1['eve_mins'], color='red', label='Evening Minutes', kde=True)

# Add legend and labels
plt.legend()
plt.xlabel('Minutes')
plt.ylabel('Frequency')
plt.title('Comparison of Day vs Evening Minutes')
plt.show()


# In[45]:


plt.figure(figsize=(10, 6))

# Plot histograms
sns.histplot(Telecom_churn_data1['day_mins'], color='blue', label='Day Minutes', kde=True)
sns.histplot(Telecom_churn_data1['night_mins'], color='green', label='Night Minutes', kde=True)

# Add legend and labels
plt.legend()
plt.xlabel('Minutes')
plt.ylabel('Frequency')
plt.title('Comparison of Day vs Night Minutes')
plt.show()


# In[46]:


plt.figure(figsize=(10, 6))

# Plot histograms
sns.histplot(Telecom_churn_data1['day_mins'], color='blue', label='Day Minutes', kde=True)
sns.histplot(Telecom_churn_data1['eve_mins'], color='orange', label='Evening Minutes', kde=True)

# Add legend and labels
plt.legend()
plt.xlabel('Minutes')
plt.ylabel('Frequency')
plt.title('Comparison of Day vs Evening Minutes')
plt.show()


# # Count Plot

# In[47]:



# Create a figure and axes
fig, axes = plt.subplots(5, 1, figsize=(8, 10))

# Plot count plot for 'churn'
sns.countplot(x='churn', data=Telecom_churn_data1, ax=axes[0])
axes[0].set_title('Count Plot for churn')

# Plot count plot for 'intl_calls'
sns.countplot(x='intl_calls', data=Telecom_churn_data1, ax=axes[1])
axes[1].set_title('Count Plot for International Calls')

# Plot count plot for 'customer_calls'
sns.countplot(x='customer_calls', data=Telecom_churn_data1, ax=axes[2])
axes[2].set_title('Count Plot for customer calls')

# Plot count plot for 'voice plan'
sns.countplot(x='voice_plan', data=Telecom_churn_data1, ax=axes[3])
axes[3].set_title('Count Plot for voice plan')

# Plot count plot for 'intl plan'
sns.countplot(x='intl_plan', data=Telecom_churn_data1, ax=axes[4])
axes[4].set_title('Count Plot for International Plan')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# # Pair plot

# In[48]:


# Pair Plot
sns.pairplot(Telecom_churn_data1, diag_kind='kde')
plt.show()


# # Heat Map

# In[49]:



# Compute the correlation matrix
correlation_matrix = Telecom_churn_data1.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # Dealing with Outliers

# # Isolation Forest

# In[50]:


Telecom_churn_data1.head(5)


# In[51]:


# training the model
clf = IsolationForest(random_state=10,contamination=.01)
clf.fit(Telecom_churn_data1)


# In[52]:


# predictions
y_pred = clf.predict(Telecom_churn_data1)

y_pred


# In[53]:


Telecom_churn_data1.loc[y_pred==-1]


# In[54]:


Telecom_churn_data1['scores']=clf.decision_function(Telecom_churn_data1)


# In[55]:


Telecom_churn_data1.shape


# In[56]:


Telecom_churn_data1['anomaly']=clf.predict(Telecom_churn_data1.iloc[:,0:20])


# In[57]:


Telecom_churn_data1


# In[58]:


#Print the outlier data points
Telecom_churn_data1[Telecom_churn_data1['anomaly']==-1]


# In[59]:


# Drop rows where 'anomaly' column is -1
Telecom_churn_data2 = Telecom_churn_data1[Telecom_churn_data1['anomaly'] != -1].copy()

# Print the cleaned DataFrame
print(Telecom_churn_data2)


# In[60]:


Telecom_churn_data2.shape


# In[61]:


Telecom_churn_data2.drop(columns=['scores', 'anomaly'], inplace=True)


# In[62]:



Telecom_churn_data2.isnull().sum()


# In[63]:



# Compute the correlation matrix
correlation_matrix = Telecom_churn_data2.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[64]:


Telecom_churn_data2.isnull().shape


# # Data Standardization

# In[65]:




# Separate features (X) and target variable (Y)
X = Telecom_churn_data2.drop('churn', axis=1)
Y = Telecom_churn_data2['churn']

# Ensure X and Y have the same number of rows
X = X[:Y.shape[0]]

# Standardize features in X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame with the standardized features
Telecom_churn_data2_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Add the target variable 'churn' to the DataFrame
Telecom_churn_data2_scaled['churn'] = Y.values

# Now Telecom_churn_data2_scaled contains the standardized features and the target variable


# In[66]:


Telecom_churn_data2_scaled.head(2)


# In[67]:


Telecom_churn_data2_scaled.shape


# In[68]:


Telecom_churn_data2_scaled.isnull().sum()


# In[69]:


Y.shape


# In[ ]:





# # Feature Engineering

# # Feature Importance using Decision Tree

# In[70]:



# Drop the target variable 'Churn' from the features
X = Telecom_churn_data2_scaled.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data2_scaled['churn']

# Feature extraction using Extra Trees Classifier
model = ExtraTreesClassifier()
model.fit(X, Y)

# Printing feature importances
print(model.feature_importances_)


# Top 7 Features 
# 
# 1.day_mins
# 2.day_charge
# 3.customer_calls
# 4.eve_mins
# 5.eve_charge
# 6..intl_calls
# 7.intl_plan
# 8.intl_mins

# In[71]:


Telecom_churn_data2_scaled.head(2)


# # Recursive Feature Elimination

# In[203]:



# Drop the target variable 'Churn' from the features
X = Telecom_churn_data2_scaled.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data2_scaled['churn']

# feature extraction
model = LogisticRegression(max_iter=400)


# In[204]:


rfe = RFE(model, n_features_to_select=8, step=1)
fit = rfe.fit(X, Y)


# In[205]:


#Num Features: 
fit.n_features_


# In[206]:


#Selected Features:
fit.support_


# In[207]:


# Feature Ranking:
fit.ranking_


# In[208]:


Telecom_churn_data2.head()


# top 7. columns are 
# 1.voice_plan 2.voice_messages 3. intl_plan 4.intl_charge 5.day_mins 6.day_charge 7.eve_charge 8. customer_calls

# In[210]:


# Selecting the desired columns
selected_columns = [
    'voice_plan', 'voice_messages', 'intl_plan','intl_charge','intl_mins', 'day_mins', 'day_charge',
    'eve_charge', 'customer_calls', 'eve_mins', 'intl_calls','churn']

# Creating a new DataFrame with the selected columns
Telecom_churn_data3 = Telecom_churn_data2_scaled[selected_columns]


# In[212]:


Telecom_churn_data3.head(2)


# In[213]:


Telecom_churn_data3.isnull().sum()


# # Model Validation Method

# # Train test split method

# In[214]:




# Drop the target variable 'Churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data3['churn']

test_size = 0.25
seed = 5
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)


result = model.score(X_test, Y_test)

result*100.0


# # K fold Cross Validation

# In[215]:




# Drop the target variable 'Churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data3['churn']

kf = KFold(n_splits=10)

mod = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), SVC()]

for i in range(len(mod)):
    kf_scores = cross_val_score(mod[i],X,Y,cv=kf)
    print('Creating Model With ' + str(mod[i]))
    print('Testing Accuracy of are Model is : ' + str(kf_scores.mean()))
    print('Printing All Training Model Accuracy : \n' + str(kf_scores))
    print('\n')


# # Model Building

# # Importing Packages

# In[216]:


Telecom_churn_data3.head(2)


# # LogisticRegression

# In[217]:


# Drop the target variable 'Churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data3['churn']


# In[218]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[219]:


# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)


# In[220]:


Telecom_churn_data3.head(5)


# In[221]:


# Predict on the testing data
y_pred = model.predict(X_test)
y_pred


# In[222]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[223]:


# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[224]:



# Make predictions on test data
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# In[225]:



# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)


# In[226]:


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# # K-Nearest Neighbour

# In[227]:


# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],       # Number of neighbors to use
    'weights': ['uniform', 'distance'],# Weight function used in prediction
    'p': [1, 2]                         # Power parameter for the Minkowski metric
}


# In[228]:


# Instantiate the KNN classifier
knn_classifier = KNeighborsClassifier()


# In[229]:


# Instantiate GridSearchCV
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)


# In[230]:


# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# In[231]:


# Get the best estimator
best_estimator = grid_search.best_estimator_


# In[232]:


# Use the best estimator to make predictions
y_pred = best_estimator.predict(X_test)


# In[233]:


# Print classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# # Support Vector Machine

# In[234]:



# Initialize the SVM model
svm_model = SVC(kernel='rbf')

# Train the SVM model on the training data
svm_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[235]:



# Initialize the SVM model
svm_model1 = SVC(kernel='linear')

# Train the SVM model on the training data
svm_model1.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm_model1.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[236]:


# Initialize the SVM model
svm_model2 = SVC(kernel='poly')

# Train the SVM model on the training data
svm_model2.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm_model2.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[237]:


# Initialize the SVM model
svm_model3 = SVC(kernel='sigmoid')

# Train the SVM model on the training data
svm_model3.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm_model3.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# # Naive Bayes 

# In[239]:


# Instantiate the Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Train the classifier on the training data
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# # Decision Tree Model

# In[240]:



# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate the decision tree classifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')

# Instantiate GridSearchCV
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = grid_search.best_estimator_

# Use the best estimator to make predictions
y_pred = best_estimator.predict(X_test)

# Print classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# # Bagging Classifier

# In[241]:


from sklearn.ensemble import BaggingClassifier

# Initialize the base classifier (e.g., Decision Tree)
base_classifier = DecisionTreeClassifier()

# Initialize the BaggingClassifier
bagging_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state = 42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = bagging_classifier.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

# Print classification report
print("Classification Report:")
print(report)


# # Random Forest

# In[242]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

# Print classification report
print("Classification Report:")
print(report)


# # Adaboost 

# In[243]:


from sklearn.ensemble import AdaBoostClassifier

# Initialize the base classifier (e.g., Decision Tree)
base_classifier = DecisionTreeClassifier(max_depth=4)  # Stump (weak learner)

# Initialize the AdaBoost classifier
ada_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=100, random_state=42)

# Train the AdaBoost classifier
ada_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = ada_classifier.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

# Print classification report
print("Classification Report:")
print(report)


# # Voting Ensemble For Classification

# In[244]:


# Voting Ensemble for Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


kfold = KFold(n_splits=10)

# create the sub models
estimators = []
model1 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = RandomForestClassifier()
estimators.append(('Random Forest', model4))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


# # Comparing Algorithms

# In[245]:


# Compare Algorithms
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Drop the target variable 'Churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data3['churn']

# prepare models
models = []
models.append(('LR', LogisticRegression(max_iter=400)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Random Forest', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=20)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# We can clearly Random forest will give more accuracy

# In[246]:


# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # ML Pipeline

# In[247]:


# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Drop the target variable 'Churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data3['churn']

# create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('Random Forest', RandomForestClassifier(n_estimators=120, random_state=42)))
model = Pipeline(estimators)

# evaluate pipeline
kfold = KFold(n_splits=10)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[248]:


# Create a pipeline that extracts features from the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# Drop the target variable 'Churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'Churn'
Y = Telecom_churn_data3['churn']

# create feature union
features = []
features.append(('pca', PCA(n_components=11)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('Random Forest', RandomForestClassifier(n_estimators=120, random_state=42)))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# # Saving Model

# In[253]:


# Save Model Using Pickle and load and predict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pickle import dump
from pickle import load

# Drop the target variable 'churn' from the features
X = Telecom_churn_data3.drop('churn', axis=1)

# Extract the target variable 'churn'
Y = Telecom_churn_data3['churn']

# Split the data into training and testing sets with a fixed random state
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, Y_train)

# Save the model to disk
# please give your local computer path to run the code 
filename = 'https://github.com/pawanmudamala/Data-Science-Projects/blob/main/Telecom%20Churn%20Project/Telecom%20churn%20Project%20code.py\\finalized_model.sav' 

dump(model, open(filename, 'wb'))

# Load the model from disk
loaded_model = load(open(filename, 'rb'))

# Evaluate the model on the test set
result = loaded_model.score(X_test, Y_test)
print("Accuracy:", result)


# In[254]:


Telecom_churn_data3.head(2)


# In[255]:


Telecom_churn_data.head(2)


# In[ ]:




