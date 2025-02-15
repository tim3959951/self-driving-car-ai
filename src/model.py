
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB7, EfficientNetB5
from tensorflow.keras.optimizers import Adam

def create_model(model_name):
    """Creates an EfficientNet-based model for angle and speed prediction."""
    if model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=(240, 320, 3))
    elif model_name == 'EfficientNetB5':
        base_model = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(240, 320, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    angle_output = Dense(1, name='angle_output')(x)
    speed_output = Dense(1, name='speed_output')(x)

    model = Model(inputs=base_model.input, outputs=[angle_output, speed_output])
    
    optimizer = Adam(learning_rate=0.001)
    loss = {'angle_output': 'mean_squared_error', 'speed_output': 'mean_squared_error'}
    metrics = {'angle_output': 'mae', 'speed_output': 'mae'}
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model
