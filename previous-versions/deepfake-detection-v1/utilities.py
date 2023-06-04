import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import tqdm


def serialize_example(image, label):
    """
    Creates a tf.train.Example message from an image and a label.

    Args:
        image (np.ndarray): The input image.
        label (np.ndarray or int): The label for the image.

    Returns:
        tf.train.Example: The Example message.
    """
    # Ensure that the label is an integer
    if isinstance(label, np.ndarray):
        print(f"Label before processing: {label}")  # print the label before processing
        if label.size > 1:
            label = np.argmax(
                label
            )  # Change this line if your labels are not one-hot encoded
        else:
            label = label[0]

    print(f"Label after processing: {label}")  # print the label after processing

    # Create a dictionary with features that may be relevant.
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.tostring()])
        ),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def write_tfrecord(dataset, filepath):
    """
    Writes a TensorFlow dataset to a TFRecord file.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset.
        filepath (str): The path to the output TFRecord file.
    """
    # Initialize a TFRecord writer
    writer = tf.io.TFRecordWriter(filepath)

    # Create a progress bar
    pbar = tqdm(total=len(dataset))

    # Loop over all records in the dataset
    for images, labels in dataset:
        # Loop over all images and labels in the batch
        for image, label in zip(images, labels):
            # Serialize the record to a string
            example = serialize_example(image.numpy(), int(label.numpy()))

            # Write the serialized string to the file
            writer.write(example)

        # Update the progress bar
        pbar.update(1)

    # Close the TFRecord writer
    writer.close()

    # Close the progress bar
    pbar.close()


def convert_gen_to_ds(gen, img_height, img_width):
    """
    Converts a data generator to a TensorFlow dataset.

    Args:
        gen (generator): The data generator to convert.

    Returns:
        tf.data.Dataset: The converted TensorFlow dataset.
    """
    return tf.data.Dataset.from_generator(
        lambda: gen,
        output_signature=(
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )


def convert_gens_to_ds(gen_list):
    """
    Converts a list of data generators to a concatenated TensorFlow dataset.

    Args:
        gen_list (list): The list of data generators to convert.

    Returns:
        tf.data.Dataset: The concatenated TensorFlow dataset.
    """
    ds = convert_gen_to_ds(gen_list[0], img_height, img_width)
    for gen in gen_list[1:]:
        ds = ds.concatenate(convert_gen_to_ds(gen, img_height, img_width))
    return ds


def generate_filename(filename):
    """
    Generates a filename with a timestamp.

    Args:
        filename (str): The original filename.

    Returns:
        str: The new filename.
    """
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    return f"{timestamp}_{filename}"
