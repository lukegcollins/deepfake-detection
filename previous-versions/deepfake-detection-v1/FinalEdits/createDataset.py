import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import xception
from tensorflow.data.Dataset import save


def create_train_val_datasets(
    preprocessing_model,
    input_dir,
    output_dir="data/datasets",
    img_height=299,
    img_width=299,
    batch_size=32,
    save_dataset=False,
    augment_training_data=False,
    preprocessing_model=None,
):
    """
    Creates train and validation datasets from images in the input directory.

    Args:
        input_dir (str): The path to the input directory containing 'real' and 'fake' subdirectories.
        output_dir (str): The directory where the TFRecord files will be saved. Default is 'data/datasets'.
        img_height (int): The height of the input images. Default is 299.
        img_width (int): The width of the input images. Default is 299.
        batch_size (int): The batch size for the data generators. Default is 32.
        save_dataset (bool): Whether to save the datasets as TFRecord files. Default is False.
        augment_training_data (bool): Whether to augment the training data. Default is False.
        preprocessing_model (callable): The preprocessing model or function to apply to the images. Default is Xception.

    Returns:
        tf.data.Dataset, tf.data.Dataset: The train and validation datasets.
    """

    # Assign Default Preprocessing Model
    if preprocessing_model is None:
        preprocessing_model = xception.preprocess_input

    # Create Training ImageDataGenerator
    # If augment_training_data is True, augment the training data
    if augment_training_data:
        datagen_train = ImageDataGenerator(
            preprocessing_function=preprocessing_model,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.2,
        )
    else:
        datagen_train = ImageDataGenerator(
            preprocessing_function=preprocessing_model, validation_split=0.2
        )

    # Create Validation ImageDataGenerator
    datagen_val = ImageDataGenerator(
        preprocessing_function=preprocessing_model, validation_split=0.2
    )

    # Create training generator
    train_gen = datagen_train.flow_from_directory(
        input_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    # Create validation data generator
    val_gen = datagen_val.flow_from_directory(
        input_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=True,
    )

    # Save the datasets if requested
    if save_dataset:
        print("Saving datasets...")
        save(train_gen, path)
        save(val_gen, path)

    # Return the datasets
    return train_gen, val_gen


def create_test_dataset(
    preprocessing_model,
    input_dir,
    output_dir="data/datasets",
    img_height=299,
    img_width=299,
    batch_size=32,
    save_dataset=False,
):
    """
    Creates a test dataset from images in the input directory.

    Args:
        input_dir (str): The path to the input directory containing 'real' and 'fake' subdirectories.
        output_dir (str): The directory where the TFRecord file will be saved. Default is 'data/datasets'.
        img_height (int): The height of the input images. Default is 299.
        img_width (int): The width of the input images. Default is 299.
        batch_size (int): The batch size for the data generators. Default is 32.
        save_dataset (bool): Whether to save the dataset as a TFRecord file. Default is False.
        preprocessing_model (callable): The preprocessing model or function to apply to the images. Default is Xception.

    Returns:
        tf.data.Dataset: The test dataset.
    """

    # Assign Default Preprocessing Model
    if preprocessing_model is None:
        preprocessing_model = xception.preprocess_input

    # Create ImageDataGenerator
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_model)

    # Create test data generator
    test_gen = datagen.flow_from_directory(
        input_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    # Save the dataset if requested
    if save_dataset:
        print("Saving datasets...")
        # Not Yet Implemented

    # Return the dataset
    return test_gen
