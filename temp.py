import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import timedelta
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("data/dhaka_temperatures_2023_2025.xlsx")
df['Date'] = pd.to_datetime(df['Date']).dt.date  # keep only date
# | Date       | AvgTemperature |
# | ---------- | -------------- |
# | 2023-01-01 | 23.5           |
# | 2023-01-02 | 24.0           |
# | ...        | ...            |

# Feature
df['Day'] = pd.to_datetime(df['Date']).apply(lambda x: x.day)
df['Month'] = pd.to_datetime(df['Date']).apply(lambda x: x.month)
df['Year'] = pd.to_datetime(df['Date']).apply(lambda x: x.year)
df['Weekday'] = pd.to_datetime(df['Date']).apply(lambda x: x.weekday())
# | Date       | AvgTemperature | Day | Month | Year | Weekday |
# | ---------- | -------------- | --- | ----- | ---- | ------- |
# | 2023-01-01 | 23.5           | 1   | 1     | 2023 | 0       |
# | 2023-01-02 | 24.0           | 2   | 1     | 2023 | 1       |
# | ...        | ...            | ... | ...   | ...  | ...     |

#features and target for regression
X = df[['Year','Month','Day','Weekday']]
y_reg = df['Temperature']

# Regression model _r,y_reg= regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
    )

reg_models = {
    "Linear Regression": LinearRegression(), #linear relationship
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42), # non-linear relationships -> n_estimators=100 → 100 trees in the forest.
    "KNN Regressor": KNeighborsRegressor() # Predicts temperature based on nearest neighbors in the feature space
}
#RMSE is a metric used to measure how far the model’s predictions are from the actual values.
# Lower RMSE = better regression model.
best_reg_rmse = float('inf')
best_reg_model_name = None
best_reg_model = None

print("=== Regression Model Evaluation ===")
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    print(f"{name} RMSE: {rmse:.2f} °C")
    
    if rmse < best_reg_rmse:
        best_reg_rmse = rmse
        best_reg_model_name = name
        best_reg_model = model

print(f"\nBest Regression Model: {best_reg_model_name} (RMSE: {best_reg_rmse:.2f} °C)")

# Classification model
def temp_category(temp):
    if temp >= 29:
        return "Hot"
    elif temp <= 21:
        return "Cold"
    else:
        return "Normal"
    
df['TempCategory'] = df['Temperature'].apply(temp_category) #apply() applies it to every row in AvgTemperature.
#LabelEncoder converts categories (Cold, Normal, Hot) into numbers the models can understand.
le = LabelEncoder()
df['TempCategoryEnc'] = le.fit_transform(df['TempCategory']) #fit_transform() → fits the encoder and transforms the column in one step.

y_clf = df['TempCategoryEnc'] #y_clf is the target variable for classification.

#_c,y_clf = classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
    )

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}
#Initialize variables to store the best model
best_clf_acc = 0
best_clf_model_name = None
best_clf_model = None

print("\nClassification Model Evaluation")
for name, model in clf_models.items(): #clf_models.items() gives you pairs of (key, value) from the dictionary
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")
    
    if acc > best_clf_acc:
        best_clf_acc = acc
        best_clf_model_name = name
        best_clf_model = model

print(f"\nBest Classification Model: {best_clf_model_name} (Accuracy: {best_clf_acc:.2f})")

# Predict next 6 months (~180 days)
last_date = df['Date'].max() #get the last date
# Generate future dates for next 6 months
future_dates = [last_date + timedelta(days=i) for i in range(1, 181)] # timedelta(days=i) → adds i days to last_date

future_df = pd.DataFrame({
    'Date': [d for d in future_dates], # extracts day of month for all future dates
    'Day': [d.day for d in future_dates],
    'Month': [d.month for d in future_dates],
    'Year': [d.year for d in future_dates],
    'Weekday': [d.weekday() for d in future_dates]
})

future_df['PredictedTemp'] = best_reg_model.predict(future_df[['Year','Month','Day','Weekday']])
future_df['Date'] = future_df['Date'].apply(lambda x: x)
# | Date       | Day | Month | Year | Weekday | PredictedTemp |
# | ---------- | --- | ----- | ---- | ------- | ------------- |
# | 2025-12-01 | 1   | 12    | 2025 | 0       | 23.6          |
# | 2025-12-02 | 2   | 12    | 2025 | 1       | 23.5          |

# Categorize next 6 months
# Convert predicted temperature into a category
future_df['Category'] = future_df['PredictedTemp'].apply(temp_category)
# Define colors for each category
color_map = {"Cold": "blue", "Normal": "green", "Hot": "red"}
# Assign colors based on category
future_df['Color'] = future_df['Category'].map(color_map)

# Plot next 6 months

plt.figure(figsize=(15,5))
# Draw the bar chart
plt.bar(future_df['Date'], future_df['PredictedTemp'], color=future_df['Color'])
plt.xlabel("Date")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Next 6 Months Predicted Temperature with Categories")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save future predictions
future_df[['Date','PredictedTemp','Category']].to_excel("next_6_months_temperature_categorized.xlsx", index=False)
print("\n 6 months predictions saved to 'next_6_months_temperature_categorized.xlsx'")

# Plot best regression model actual vs predicted
y_pred_best = best_reg_model.predict(X_test_r)
plt.figure(figsize=(10,5))
plt.scatter(y_test_r, y_pred_best, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title(f"{best_reg_model_name}: Actual vs Predicted")
plt.show()
