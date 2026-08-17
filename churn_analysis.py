# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Create output folder

# This is important because you want to save your charts like you did in your previous project.
import os
os.makedirs("output", exist_ok=True)

# Load the dataset
dataset = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# print(dataset.head())
# Understand the dataset
# print(dataset.shape)
# print(dataset.info())
# print(dataset.describe())
# print(dataset.isnull().sum())
# print(dataset.columns)
# Data Cleaning...........................................................

dataset["TotalCharges"] = pd.to_numeric( dataset["TotalCharges"],errors="coerce")
# print(dataset.isnull().sum()) #Check missing values
dataset.dropna(inplace=True) #Remove the missing values
# print(dataset.isnull().sum()) #Verify that there are no more missing values
dataset.drop("customerID", axis=1, inplace=True) #Remove unnecessary customer ID

# Churn distribution...................................................................

# print(dataset["Churn"].value_counts())

# Create figure
# plt.figure(figsize=(9, 6))


# # Different colors for No and Yes
# colors = ["#2E86DE", "#E74C3C"]

# sns.countplot(
#     x="Churn",
#     data=dataset,
#     hue="Churn", #This tells Seaborn to use the Churn categories to separate/color the bars.
#     palette=colors,
#     legend=False
# )

# # Title and labels
# plt.title(
#     "Customer Churn Distribution",
#     fontsize=18,
#     fontweight="bold",
#     pad=20 # Adds some space between the title and the chart.
# )

# plt.xlabel("Churn Status", fontsize=12, fontweight="bold")
# plt.ylabel("Number of Customers", fontsize=12, fontweight="bold")

# # Improve x-axis labels
# plt.xticks(
#     [0, 1],
#     ["No – Stayed", "Yes – Churned"],
#     fontsize=11
# )

# # Add light grid
# plt.grid(
#     axis="y",
#     linestyle="--",
#     alpha=1
# )

# # Remove unnecessary borders
# sns.despine()

# # Adjust layout
# plt.tight_layout()

# # Save high-quality PNG
# plt.savefig(
#     "output/churn_distribution.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# # Calculate churn percentage
# churn_percentage = (
#     dataset["Churn"]
#     .value_counts(normalize=True) * 100
# )

# print(churn_percentage)

# Analyze churn by contract.............................
# contract_churn = pd.crosstab(
#     dataset["Contract"],
#     dataset["Churn"],
#     normalize="index"
# ) * 100

# print(contract_churn)

# contract_churn.plot(
#     kind="bar",
#     figsize=(10, 6),
#     color=["green","red"]
# )

# plt.title("Churn Rate by Contract Type")
# plt.xlabel("Contract Type")
# plt.ylabel("Percentage")
# plt.xticks(rotation=0)
# plt.legend(title="Churn")
# plt.grid(
#     axis="y",
#     linestyle="--",
#     alpha=1
# )

# plt.savefig(
#     "output/churn_by_contract.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
# Churn by Internet Service..................................
# dataset["InternetService"]=dataset["InternetService"].replace("No","No Internet Service")
# internet_churn = pd.crosstab(
#     dataset["InternetService"],
#     dataset["Churn"],
#     normalize="index"
# ) * 100

# print(internet_churn)
# internet_churn.plot(
#     kind="bar",
#     figsize=(10, 6)
# )

# plt.title("Churn Rate by Internet Service")
# plt.xlabel("Internet Service")
# plt.ylabel("Percentage")
# plt.xticks(rotation=0)
# plt.grid(
#     axis="y",
#     linestyle="--",
#     alpha=1
# )


# plt.savefig(
#     "output/churn_by_internet_service.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# Churn by Payment Method........................
# payment_churn = pd.crosstab(
#     dataset["PaymentMethod"],
#     dataset["Churn"],
#     normalize="index"
# ) * 100

# print(payment_churn)
# payment_churn.plot(
#     kind="bar",
#     figsize=(12, 6),
#     color=["pink","yellow"]

# )

# plt.title("Churn Rate by Payment Method")
# plt.xlabel("Payment Method",fontsize=12,fontweight="bold")
# plt.ylabel("Percentage",fontsize=12,fontweight="bold")
# plt.xticks(rotation=0)

# plt.savefig(
#     "output/churn_by_payment_method.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# Churn by tenure........................
# dataset["tenure_group"] = pd.cut(
#     dataset["tenure"],
#     bins=[0, 12, 24, 48, 72],
#     labels=[
#         "0-12 Months",
#         "13-24 Months",
#         "25-48 Months",
#         "49-72 Months"
#     ]
# )
# tenure_churn = pd.crosstab(
#     dataset["tenure_group"],
#     dataset["Churn"],
#     normalize="index"
# ) * 100

# print(tenure_churn)

# tenure_churn.plot(
#     kind="bar",
#     figsize=(10, 6)
# )

# plt.title("Churn Rate by Customer Tenure")
# plt.xlabel("Tenure")
# plt.ylabel("Percentage")
# plt.xticks(rotation=0)

# plt.savefig(
#     "output/churn_by_tenure.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
# Monthly Charges vs Churn............................
# plt.figure(figsize=(8, 6))

# sns.boxplot(
#     x="Churn",
#     y="MonthlyCharges",
#     palette=["skyblue","orange"],
#     data=dataset
# )

# plt.title("Monthly Charges by Churn Status")
# plt.xlabel("Churn")
# plt.ylabel("Monthly Charges")

# plt.savefig(
#     "output/monthly_charges_churn.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# Correlation Heatmap.........................
# numeric_data = dataset.select_dtypes(
#     include=np.number
# )
# plt.figure(figsize=(12, 8))

# sns.heatmap(
#     numeric_data.corr(),
#     annot=True,
#     cmap="coolwarm"
# )

# plt.title("Correlation Heatmap")

# plt.savefig(
#     "output/correlation_heatmap.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# Add Machine Learning Prediction ⭐
# Categorical Variables → Numbers
# A machine-learning model works with numerical values, so we convert these categories into numbers.
from sklearn.preprocessing import LabelEncoder #Imports the LabelEncoder tool from Scikit-learn.
labelencoder = LabelEncoder()  #Creates a LabelEncoder object.
categorical_cols = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'Churn'
]
for col in categorical_cols:
    dataset[col] = labelencoder.fit_transform(dataset[col]) # Go through each column in the list and convert its categories into numbers.

# Feature Selection

# Now we decide what information the model should use.

X = dataset.drop(['Churn'], axis=1)

# X contains the input features.
# We remove:customerID
# Churn: We remove it because Churn is the answer we want the model to predict.
# X = information about customer

# Target Variable
y = dataset['Churn']
# y contains the answer/target.
# Train-Test Split....................................
from sklearn.model_selection import train_test_split #Imports the function used to divide your dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0  #Makes the split repeatable. If you run the program again, you get the same split.
)

# Feature Scaling
# Feature Scaling means bringing different numerical features 
# to a similar scale so that one feature doesn't dominate because it has larger values.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# This does two things:Learn the scaling rules from the training data and then apply those rules to the training data.
#fit → learn the rules
# transform → apply the rules
# The resulting values will generally be centered around:
# Mean ≈ 0
# Standard deviation ≈ 1

X_test = scaler.transform(X_test)  #This scales the test data using the same scaling information learned from X_train.


# Model Training and Prediction......................
from sklearn.ensemble import RandomForestClassifier

# What is Random Forest?

# Random Forest is a machine-learning algorithm that uses many decision trees together.

# Think of it like asking many people for their opinion:

# Decision Tree 1 → Churn
# Decision Tree 2 → No Churn
# Decision Tree 3 → Churn
# Decision Tree 4 → Churn
# Decision Tree 5 → Churn


#               ↓


#        Majority decision
#               ↓
#            Churn

# Instead of depending on one decision tree, Random Forest combines many trees to make a more reliable prediction.
clf = RandomForestClassifier()
# Train the model...........................
clf.fit(X_train, y_train)
# This is where the model learns from your training data.
# X_train → Customer information
# y_train → Actual Churn result
# Make predictions...........................
y_pred = clf.predict(X_test) #Now the trained model receives the test customers and predicts their churn status.

# Accuracy Score......................................
# Now we want to know: How many predictions did the model get correct?
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
# This compares:
# y_test → Actual answers
# y_pred → Model predictions
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix....................................................
# What the confusion matrix means
# You will generally see something like:

# Actual / Predicted	No Churn	Churn......#
#   No Churn	            924	     117       #
#    Churn	                187	     181       #
#..............................................#
# Accuracy alone isn't enough for a churn project.
# We also want to know: Which customers did the model correctly or incorrectly classify?

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Two tools are imported:
# confusion_matrix → calculates the actual vs predicted results.
# ConfusionMatrixDisplay → displays those results as a chart.
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["No Churn", "Churn"]
# )
# disp.plot(cmap="coolwarm")
# This creates the visual confusion matrix.
# cmap="coolwarm" controls the color scheme of the matrix.
# plt.title("Confusion Matrix")
# plt.tight_layout()

# plt.savefig(
#     "output/confusion_matrix.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

# Easy way to remember
#                   MODEL PREDICTION
#                  No Churn    Churn
#               ┌───────────┬─────────┐
# ACTUAL        │           │         │
# No Churn      │    TN     │   FP    │
#               ├───────────┼─────────┤
# Churn         │    FN     │   TP    │
#               └───────────┴─────────┘

# Where:
# TN = Correctly predicted No Churn
# TP = Correctly predicted Churn
# FP = Wrongly predicted Churn
# FN = Missed Churn

# Classification Report.....................
# A classification report is used to check how well a classification machine-learning model is performing.
# In your project, the model is predicting:
# 0 → Customer did not churn
# 1 → Customer churned

# The classification report gives you four important things:
# Precision
# Recall
# F1-score
# Support

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

# sklearn.metrics contains functions for measuring model performance.
# classification_report is one of those functions.

# What is Feature Importance?.......................................

# Your dataset has many features:
# gender
# SeniorCitizen
# tenure
# InternetService
# OnlineSecurity
# Contract
# MonthlyCharges
# TotalCharges
# PaymentMethod
# ...

# The Random Forest model looks at these features while making predictions.

# Feature importance tells us which features the model relied on more when making its predictions.

# For example, you might get:

# tenure            0.25
# MonthlyCharges    0.18
# TotalCharges      0.15
# Contract          0.13
# InternetService   0.10
# ...

# This means tenure was more important to the Random Forest model than InternetService.

# ⚠️ Important: Feature importance does not mean that the feature causes churn.
#  It means the feature was useful to the model for making predictions.
feature_importance =pd.Series(                      #pd.Series() creates a Pandas Series.
    clf.feature_importances_,   #might give something like:[0.12, 0.05, 0.25, 0.18, 0.10, ...]
    index=X.columns
).sort_values(ascending=False)

# The numbers from feature_importances_ need feature names.
# Your X.columns contains:
# gender
# SeniorCitizen
# Partner
# Dependents
# tenure
# PhoneService
# ...
# MonthlyCharges
# TotalCharges
plt.style.use("dark_background")
plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance.values,
    y=feature_importance.index,
    color="red",
    edgecolor="white",
    linewidth=1
)
# This creates the graph using Seaborn.
# x=feature_importance.values
# These are the importance numbers:
# 0.25
# 0.18
# 0.13
# 0.10
# ...
# They go on the X-axis.

# y=feature_importance.index
# These are the feature names:
# tenure
# MonthlyCharges
# Contract
# InternetService
# They go on the Y-axis.
plt.title("Top Factors Influencing Customer Churn",fontsize=15,fontweight="bold",color="yellow")
plt.xlabel("Importance",fontsize=12,fontweight="bold")
plt.ylabel("Features",fontsize=12,fontweight="bold")
plt.tight_layout()
plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.5
)
plt.savefig(
    "output/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()