import tensorflow as tf
from data_loader import DataGenerator,designate_batches



def train_model_generator(model,train_gen, valid_gen, steps_per_epoch, checkpoint_filepath, validation_steps,
                          epochs=100, batch_size=32, verbose=1):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    callbacks = [model_checkpoint_callback]

    history = model.fit(
        train_gen,  # Training generator
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,  # Validation generator
        validation_steps=validation_steps,  # Number of validation steps
        callbacks=callbacks
    )
    return history


# Define your model architecture
def build_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    #r = tf.keras.layers.Rescaling(scale=1. / 255.)(inputs)

    # Encoder
    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.15)(c1)
    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.15)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.22)(c3)
    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.22)(c4)
    c4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c9)

    # Output
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Define input shape based on your image size
input_shape = (560,960,3)

# Compile the model
model = build_model(input_shape)



if tf.test.is_gpu_available():
    print("GPU is available. You're using GPU for training.")
else:
    print("GPU is not available. You're using CPU for training.")


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

batch_size=4
train_batches=designate_batches("/Desktop/cnn/","train",batch_size)
valid_batches=designate_batches("/Desktop/cnn/","valid",batch_size)
test_batches=designate_batches("/Desktop/cnn/","test",batch_size)
train_dt=DataGenerator(train_batches,batch_size)
valid_dt=DataGenerator(valid_batches,batch_size)

train_model_generator(model,train_dt,valid_dt,len(train_batches),checkpoint_filepath="/Desktop/cnn/models",epochs=25,batch_size=batch_size,validation_steps=len(valid_batches))