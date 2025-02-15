
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# Load trained models
model_files = ["model_0.h5", "model_1.h5"]
models = [load_model(model_file) for model_file in model_files]

# Load test images
def load_test_images(num_images, base_path):
    images = []
    for i in range(1, num_images + 1):
        image = cv2.imread(f'{base_path}/{i}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.array(images)

test_images = load_test_images(1020, 'data/test_data/test_data')

# Perform ensemble predictions
test_predictions = [model.predict(test_images) for model in models]
test_angle_predictions = np.mean([predictions[0] for predictions in test_predictions], axis=0)
test_speed_predictions = np.mean([predictions[1] for predictions in test_predictions], axis=0)

# Save predictions
submission_df = pd.DataFrame({
    'image_id': np.arange(1, 1021),
    'angle': test_angle_predictions.flatten(),
    'speed': test_speed_predictions.flatten()
})
submission_df.to_csv('submission.csv', index=False)
