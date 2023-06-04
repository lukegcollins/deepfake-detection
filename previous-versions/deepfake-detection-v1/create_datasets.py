import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utilities import (
    serialize_example,
    write_tfrecord,
    convert_gen_to_ds,
    convert_gens_to_ds,
    generate_filename,
)


def create_train_val_datasets(
    input_dir,
    output_dir="data/datasets",
    img_height=299,
    img_width=299,
    batch_size=32,
    save_dataset=True,
):
    """
    Creates train and validation datasets from images in the input directory.

    Args:
        input_dir (str): The path to the input directory containing 'real' and 'fake' subdirectories.
        output_dir (str): The directory where the TFRecord files will be saved. Default is 'data/datasets'.
        img_height (int): The height of the input images. Default is 299.
        img_width (int): The width of the input images. Default is 299.
        batch_size (int): The batch size for the data generators. Default is 32.
        save_dataset (bool): Whether to save the datasets as TFRecord files. Default is True.

    Returns:
        tf.data.Dataset, tf.data.Dataset: The train and validation datasets.
    """
    # Create ImageDataGenerator with validation split
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.xception.preprocess_input,
        validation_split=0.2,
    )

    # Create a train and validation data generator
    train_gen = datagen.flow_from_directory(
        input_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        input_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
    )

    # Convert generators to TensorFlow Datasets
    train_ds = convert_gen_to_ds(train_gen, img_height, img_width)
    val_ds = convert_gen_to_ds(val_gen, img_height, img_width)

    # Save the dataset if requested
    if save_dataset:
        train_filename = generate_filename("train_ds")
        val_filename = generate_filename("val_ds")
        tf.data.Dataset.save(train_ds, f"{output_dir}/{train_filename}.tfrecords")
        tf.data.Dataset.save(val_ds, f"{output_dir}/{val_filename}.tfrecords")

    return train_ds, val_ds


def create_test_dataset(
    input_dir,
    output_dir="data/datasets",
    img_height=299,
    img_width=299,
    batch_size=32,
    save_dataset=True,
):
    """
    Creates a test dataset from images in the input directory.

    Args:
        input_dir (str): The path to the input directory containing 'real' and 'fake' subdirectories.
        output_dir (str): The directory where the TFRecord file will be saved. Default is 'data/datasets'.
        img_height (int): The height of the input images. Default is 299.
        img_width (int): The width of the input images. Default is 299.
        batch_size (int): The batch size for the data generators. Default is 32.
        save_dataset (bool): Whether to save the dataset as a TFRecord file. Default is True.

    Returns:
        tf.data.Dataset: The test dataset.
    """
    # Create ImageDataGenerator
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.xception.preprocess_input,
    )

    # Load all images
    test_gen = datagen.flow_from_directory(
        input_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
    )

    # Convert generator to TensorFlow Datasets
    test_ds = convert_gen_to_ds(test_gen, img_height, img_width)

    # Save the dataset if requested
    if save_dataset:
        test_filename = generate_filename("test_ds")
        tf.data.Dataset.save(test_ds, f"{output_dir}/{test_filename}.tfrecords")

        # write_tfrecord(test_ds, f"{output_dir}/{test_filename}.tfrecords")

    return test_ds
