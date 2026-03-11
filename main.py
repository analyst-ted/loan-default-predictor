from src.data_processing import load_data, check_missing_values, fill_mort_acc, final_clean, prepare_features
from src.model import split_and_scale, build_model, evaluate_model
import joblib
import numpy as np
import pandas as pd

# 1.Path to dataset
data_path = '/Users/aruproy/Documents/projects/loan_default_project/data/lending_club_loan_two.csv'

#  2.Executing the ingestion function
print('Starting Startup Loan Default Pipeline')
df = load_data(data_path)

# 3. Checking for missing values
missing = check_missing_values(df)

# 4. Cleaning the mort_acc column
df = fill_mort_acc(df)

# 5. Final Cleaning of rest of the columns
df = final_clean(df)

# 6. Checking the missing again
check_missing_values(df)

# 7. Preparing the features
df = prepare_features(df)

# 8. Splitting the data and scaling it
X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

# 9. Building the model
model = build_model(X_train.shape[1])

# 10. training the model
print('Starting Deep Learning Training')
weights = {0:4.0, 1:1.0}
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=256,
    validation_data=(X_test, y_test),
    class_weight = weights
)

#. 11 Evaluating the model

evaluate_model(model,X_test,y_test)

# --- EXPORT THE ASSETS ---
print("\nSaving model and scaler for deployment...")

# Save the Keras Deep Learning Model
print("\nSaving model, scaler, and baselines for deployment...")

model.save('loan_model.keras')
joblib.dump(scaler, 'scaler.pkl')

# NEW: Calculate the median of every column and save it as our baseline blueprint
baselines = pd.Series(np.median(X_train, axis=0), index=scaler.feature_names_in_) 
print(baselines)
joblib.dump(baselines, 'baselines.pkl')

print("Assets saved successfully! Ready for web deployment.")
