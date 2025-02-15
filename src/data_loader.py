
import numpy as np
import pandas as pd
import cv2

# Load training dataset
train_df = pd.read_csv('data/training_norm.csv')

# Load train images
def load_images(image_ids, base_path):
    images = []
    for image_id in image_ids:
        image = cv2.imread(f'{base_path}/{image_id}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.array(images)

train_images = load_images(train_df['image_id'], 'data/training_data/training_data')
train_angles = np.array(train_df['angle'])
train_speeds = np.array(train_df['speed'])
