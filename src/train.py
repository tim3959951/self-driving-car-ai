
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from model import create_model
from data_loader import train_images, train_df, train_angles, train_speeds
from augmentation import augment_images

# Learning rate scheduler
def scheduler(epoch, lr):
    return lr * 0.9 if epoch >= 10 else lr

lr_scheduler = LearningRateScheduler(scheduler)

# Apply image augmentation
train_images = augment_images(train_images)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
fold = 1
model_names = ['EfficientNetB7', 'EfficientNetB5']

for train_index, val_index in kf.split(train_images):
    X_train, X_val = train_images[train_index], train_images[val_index]
    y_train, y_val = train_df[['angle', 'speed']].values[train_index], train_df[['angle', 'speed']].values[val_index]

    for model_name in model_names:
        model = create_model(model_name)
        model_checkpoint = ModelCheckpoint(f"model_{model_name}_fold_{fold}.h5", save_best_only=True, monitor='val_loss', mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        model.fit(X_train, [y_train[:, 0], y_train[:, 1]], epochs=100, batch_size=32, validation_data=(X_val, [y_val[:, 0], y_val[:, 1]]), callbacks=[early_stop, lr_scheduler, model_checkpoint])

        models.append(model)
    fold += 1
