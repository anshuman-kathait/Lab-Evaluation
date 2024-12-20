import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('C:/Users/katha/OneDrive/Desktop/ML LAB EVAL/Fuel_cell_performance_data-Full.csv')

if 'Target5' not in data.columns:
    raise ValueError("Target5 not found in the dataset. Check column names.")

# Ensure only F1 to F15 columns and Target5 are used
data = data[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'Target5']]

#Split the dataset into features and target
X = data.drop(columns=['Target5'])
y = data['Target5']

#missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {'R2 Score': r2, 'MSE': mse}

best_model = max(results, key=lambda x: results[x]['R2 Score'])

detailed_results = pd.DataFrame(results).T

print("\nDetailed Model Evaluation:")
print(detailed_results)
print(f"\nBest Model: {best_model} with R2 Score: {results[best_model]['R2 Score']:.4f}")

output_path = 'C:/Users/katha/OneDrive/Desktop/ML LAB EVAL/model_evaluation_results.csv'
detailed_results.to_csv(output_path, index=True)
print(f"\nEvaluation results saved to {output_path}")
