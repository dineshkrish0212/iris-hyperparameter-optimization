import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import optuna.visualization as vis
import numpy as np
import plotly.io as pio

# Load Dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)  # Added stratify

# ---- OPTUNA (Bayesian) ----
def objective(trial):
    # Define search space
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
    max_features = trial.suggest_float("max_features", 0.3, 1.0)  # Limited range to reduce overfitting

    # Define Random Forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,  # Added feature control
        bootstrap=True,  # Enable bootstrapping
        random_state=42
    )

    # Cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

# Run Optuna Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Train final model with best Optuna params
best_optuna_params = study.best_params
optuna_model = RandomForestClassifier(**best_optuna_params, random_state=42)
optuna_model.fit(X_train, y_train)
optuna_acc = optuna_model.score(X_test, y_test)

# ---- GRID SEARCH ----
grid_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "max_features": [0.4, 0.6, 0.8]  # Adjusted to reduce overfitting
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_grid_params = grid_search.best_params_
grid_model = RandomForestClassifier(**best_grid_params, random_state=42)
grid_model.fit(X_train, y_train)
grid_acc = grid_model.score(X_test, y_test)

# ---- RANDOM SEARCH ----
random_params = {
    "n_estimators": np.arange(50, 200, 50),
    "max_depth": np.arange(5, 20, 5),
    "min_samples_split": np.arange(2, 20, 5),
    "min_samples_leaf": np.arange(1, 10, 2),
    "max_features": np.linspace(0.3, 0.8, 4)  # Adjusted range
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), random_params, n_iter=20, cv=5, scoring="accuracy", n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
best_random_params = random_search.best_params_
random_model = RandomForestClassifier(**best_random_params, random_state=42)
random_model.fit(X_train, y_train)
random_acc = random_model.score(X_test, y_test)

# ---- RESULTS ----
print("\n--- Accuracy Comparison ---")
print(f"Optuna (Bayesian) Accuracy: {optuna_acc:.4f}")
print(f"Grid Search Accuracy: {grid_acc:.4f}")
print(f"Random Search Accuracy: {random_acc:.4f}")

# ---- VISUALIZATION ----
pio.write_image(vis.plot_optimization_history(study), "optuna_optimization_history.png")
pio.write_image(vis.plot_param_importances(study), "optuna_param_importance.png")
