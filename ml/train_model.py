import os
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "session_data.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading rehabilitation session data...")

df = pd.read_csv(DATA_FILE)

print(f"Total sessions found: {len(df)}")


# =========================================================
# SORT SESSIONS
# =========================================================

df["DateTime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"]
)

df = df.sort_values(
    "DateTime"
).reset_index(drop=True)


# =========================================================
# CREATE NEXT-SESSION TARGET
# =========================================================

# The target is the ROM of the NEXT session.

df["Next_ROM"] = df["ROM_Range"].shift(-1)


# Previous session ROM
df["Previous_ROM"] = df["ROM_Range"].shift(1)


# Change from previous session
df["ROM_Change"] = (
    df["ROM_Range"] -
    df["Previous_ROM"]
)


# Rolling average of previous sessions
df["Previous_Average_ROM"] = (
    df["ROM_Range"]
    .shift(1)
    .rolling(window=3)
    .mean()
)


# Remove rows where the required values don't exist
ml_df = df.dropna(
    subset=[
        "Next_ROM",
        "Previous_ROM",
        "ROM_Change",
        "Previous_Average_ROM"
    ]
).copy()


print(
    f"Usable ML samples: {len(ml_df)}"
)


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "ROM_Range",
    "Previous_ROM",
    "ROM_Change",
    "Previous_Average_ROM",
    "Repetitions",
    "Correct_Repetitions",
    "Incorrect_Repetitions",
    "Accuracy",
    "Target_Angle"
]

TARGET = "Next_ROM"


X = ml_df[FEATURES]

y = ml_df[TARGET]


# =========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

# IMPORTANT:
# We don't randomly shuffle because rehabilitation
# sessions have a time sequence.

split_index = int(
    len(X) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# =========================================================
# LINEAR REGRESSION
# =========================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_prediction = linear_model.predict(
    X_test
)


linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_prediction
    )
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_prediction = rf_model.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_prediction
    )
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 55)
print("MODEL PERFORMANCE")
print("=" * 55)

print("\nLinear Regression")
print(
    f"MAE  : {linear_mae:.2f}°"
)
print(
    f"RMSE : {linear_rmse:.2f}°"
)
print(
    f"R²   : {linear_r2:.3f}"
)

print("\nRandom Forest Regression")
print(
    f"MAE  : {rf_mae:.2f}°"
)
print(
    f"RMSE : {rf_rmse:.2f}°"
)
print(
    f"R²   : {rf_r2:.3f}"
)


# =========================================================
# SELECT BEST MODEL
# =========================================================

if rf_mae < linear_mae:

    best_model = rf_model
    best_name = "Random Forest Regression"
    best_mae = rf_mae

else:

    best_model = linear_model
    best_name = "Linear Regression"
    best_mae = linear_mae


print("\n" + "=" * 55)

print(
    f"BEST MODEL: {best_name}"
)

print(
    f"Best MAE: {best_mae:.2f}°"
)

print("=" * 55)


# =========================================================
# SAVE MODEL
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "rom_prediction_model.pkl"
)

model_data = {
    "model": best_model,
    "features": FEATURES,
    "model_name": best_name
}

with open(
    model_path,
    "wb"
) as file:

    pickle.dump(
        model_data,
        file
    )


print(
    f"\nModel saved to:\n{model_path}"
)


# =========================================================
# SHOW ACTUAL VS PREDICTED
# =========================================================

results = pd.DataFrame({

    "Actual_Next_ROM": y_test.values,

    "Predicted_Next_ROM": np.round(
        best_model.predict(X_test),
        2
    )

})

print("\nActual vs Predicted ROM:")
print(results.to_string(index=False))


print("\nML training completed successfully.")