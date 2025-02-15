
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load true values from training data
train_df = pd.read_csv('data/training_norm.csv')

# Load predictions
submission_df = pd.read_csv('submission_modified.csv')

# Merge predictions with true values
merged_df = train_df.merge(submission_df, on='image_id', suffixes=('_true', '_pred'))

# Compute evaluation metrics
angle_mse = mean_squared_error(merged_df['angle_true'], merged_df['angle_pred'])
angle_mae = mean_absolute_error(merged_df['angle_true'], merged_df['angle_pred'])
angle_r2 = r2_score(merged_df['angle_true'], merged_df['angle_pred'])

speed_mse = mean_squared_error(merged_df['speed_true'], merged_df['speed_pred'])
speed_mae = mean_absolute_error(merged_df['speed_true'], merged_df['speed_pred'])
speed_r2 = r2_score(merged_df['speed_true'], merged_df['speed_pred'])

print("Angle Prediction Metrics:")
print(f"  MSE: {angle_mse:.4f}, MAE: {angle_mae:.4f}, R² Score: {angle_r2:.4f}")
print("Speed Prediction Metrics:")
print(f"  MSE: {speed_mse:.4f}, MAE: {speed_mae:.4f}, R² Score: {speed_r2:.4f}")

# Visualization: True vs Predicted values for angle
plt.figure(figsize=(12, 5))
plt.scatter(merged_df['angle_true'], merged_df['angle_pred'], alpha=0.5)
plt.plot([merged_df['angle_true'].min(), merged_df['angle_true'].max()],
         [merged_df['angle_true'].min(), merged_df['angle_true'].max()],
         color='red', linestyle='dashed')
plt.xlabel("True Angle")
plt.ylabel("Predicted Angle")
plt.title("True vs Predicted Angle")
plt.grid(True)
plt.show()

# Visualization: True vs Predicted values for speed
plt.figure(figsize=(12, 5))
plt.scatter(merged_df['speed_true'], merged_df['speed_pred'], alpha=0.5)
plt.plot([merged_df['speed_true'].min(), merged_df['speed_true'].max()],
         [merged_df['speed_true'].min(), merged_df['speed_true'].max()],
         color='red', linestyle='dashed')
plt.xlabel("True Speed")
plt.ylabel("Predicted Speed")
plt.title("True vs Predicted Speed")
plt.grid(True)
plt.show()

# Error distribution
plt.figure(figsize=(12, 5))
plt.hist(merged_df['angle_true'] - merged_df['angle_pred'], bins=30, alpha=0.7, label='Angle Error')
plt.hist(merged_df['speed_true'] - merged_df['speed_pred'], bins=30, alpha=0.7, label='Speed Error')
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.legend()
plt.grid(True)
plt.show()
