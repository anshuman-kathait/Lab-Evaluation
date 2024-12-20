import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load the dataset
data = pd.read_csv('C:/Users/katha/OneDrive/Desktop/ML LAB EVAL/Fuel_cell_performance_data-Full.csv')

# Step 2: Select Target 5 (Your roll number ends with 9)
if 'Target5' not in data.columns:
    raise ValueError("Target5 not found in the dataset. Check column names.")

data = data.drop(columns=[col for col in data.columns if col.startswith('Target') and col != 'Target5'])

# Step 3: Split the dataset into features and target
X = data.drop(columns=['Target5'])
y = data['Target5']

# Handle missing values (if any)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Step 4: Divide the dataset into 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train multiple prediction models and evaluate them
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

# Step 6: Identify the best model
best_model = max(results, key=lambda x: results[x]['R2 Score'])

# Step 7: Print detailed results and the best model
detailed_results = pd.DataFrame(results).T

print("\nDetailed Model Evaluation:")
print(detailed_results)
print(f"\nBest Model: {best_model} with R2 Score: {results[best_model]['R2 Score']:.4f}")

# Step 8: Save the results to a CSV file
output_path = 'C:/Users/katha/OneDrive/Desktop/ML LAB EVAL/model_evaluation_results.csv'
detailed_results.to_csv(output_path, index=True)
print(f"\nEvaluation results saved to {output_path}")
