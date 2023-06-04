import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
)

from create_datasets import create_test_dataset, create_train_val_datasets
from utilities import generate_filename

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)]
)

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


# # Define basemodel from Xception
# basemodel = Xception(weights = 'imagenet',
#                      include_top = False,  # Do not include the ImageNet classifier at the top
#                      input_tensor = Input(shape = (299, 299, 3)))
# # freeze all layers except last 23, weights of last 23 layers we be updated
# for layer in basemodel.layers[:-23]:
#     layer.trainable = Fals
# # Define LAyers
# headmodel = basemodel.output
# headmodel = GlobalAveragePooling2D()(headmodel)
# headmodel = Dense(2048, activation = 'relu')(headmodel)
# headmodel = Dropout(0.2)(headmodel)
# headmodel = Dense(1, activation = 'sigmoid')(headmodel)


# # Group the basemodel and new fully-connected layers into a Model object
# model = Model(inputs = basemodel.input, outputs = headmodel)


def build_base_model(augment_data=True, print_summary=True):
    # Step 1: Setup Data Augmentation
    model_input = Input(shape=(299, 299, 3))
    modified_input = data_augmentation(model_input) if augment_data else model_input

    # Step 2: Build Base Model
    base_model = keras.applications.Xception(
        weights="imagenet",
        input_tensor=modified_input,
        include_top=False,
    )

    base_model.trainable = False

    # Step 3: Build Top Model (Layers)
    top_model = base_model.output
    top_model = keras.layers.GlobalAveragePooling2D()(top_model)
    top_model = keras.layers.Dense(2048)(top_model)
    top_model = keras.layers.Dropout(0.2)(top_model)
    top_model = keras.layers.Dense(1, activation="sigmoid")(top_model)

    # Step 4: Combine Base Model and Top Model
    whole_model = Model(inputs=base_model.input, outputs=top_model)

    # Step 5: Generate Summary
    if print_summary:
        whole_model.summary()

    # Step 6: Return base_model and whole_model
    return base_model, whole_model


def train_top_layer(
    model,
    train_ds,
    validation_ds,
    epochs=10,
    save_dir="data/models",
    print_summary=True,
):
    # Step 2: Train Top Layer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=validation_ds,
        # use_multiprocessing=True,
        # workers=8,
    )

    if save_dataset:
        filename = generate_filename("top_layer")
        model.save(f"{save_dir}/{filename}.h5")

    if print_summary:
        print("Top layer Summary:")
        model.summary()

    return model


def train_entire_model(
    base_model,
    model,
    train_ds,
    validation_ds,
    epochs=10,
    print_summary=False,
    save_dataset=True,
    save_dir="data/models",
):
    # Step 3: Train Entire Model
    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    if save_dataset:
        filename = generate_filename("entire_model")
        model.save(f"{save_dir}/{filename}.h5")

    if print_summary:
        print("Entire Model Summary:")
        model.summary()

    return model


def main():
    print("Lets Detect some Deepfakes!")
    print(sys.executable)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)

    test_data = create_test_dataset(
        "/media/luke/System Storage/Deepfake Dataset/FF++/combine/test",
        save_dataset=False,
    )
    train_data, val_data = create_train_val_datasets(
        "/media/luke/System Storage/Deepfake Dataset/FF++/combine/train",
        save_dataset=False,
    )

    base_model, model = build_base_model(print_summary=True)

    trained_model = train_top_layer(model, train_data, val_data, print_summary=True)

    new_model = train_entire_model(
        base_model, trained_model, train_data, val_data, print_summary=True
    )

    print("Evaluating model...")

    results = new_model.evaluate(test_data)

    print("Test loss, Test acc:", results)


if __name__ == "__main__":
    main()
