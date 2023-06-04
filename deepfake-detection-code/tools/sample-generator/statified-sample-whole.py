import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm


def create_stratified_sample(source_directory, target_directory, sample_ratio):
    """
    Breaks down a dataset into stratified subsets according to the given sample_ratio.

    The subsets are saved in the target_directory, with each subset stored in a separate subdirectory named train-{i}.
    This function works with a two-class problem, specifically classes 'real' and 'fake'.
    The 'fake' class may contain multiple subdirectories.

    Args:
        source_directory (str): The directory of the original dataset.
        target_directory (str): The directory where the subsets will be created.
        sample_ratio (float): The ratio of the original dataset to include in each subset (e.g., 0.1 for 10%).
    """

    # Create target directory if it does not exist.
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Calculate the number of subsets and retrieve the base name of the source directory.
    subset_num = int(1 / sample_ratio)
    subset_name = os.path.basename(source_directory)

    for subset in range(subset_num):
        # Create a directory for each subset
        subset_target_directory = f"{target_directory}/train-{subset+1}"
        os.makedirs(subset_target_directory, exist_ok=True)

        for class_dir in os.listdir(source_directory):
            # Create a directory for each class in the subset directory
            os.makedirs(f"{subset_target_directory}/{class_dir}", exist_ok=True)

            if class_dir == "real":  # 'real' directory directly contains images
                # Obtain a list of files in the 'real' directory
                class_files = os.listdir(f"{source_directory}/{class_dir}")

                # Randomly shuffle the file list
                np.random.shuffle(class_files)

                # Calculate the start and end indices for the current subset
                start_index = int(len(class_files) * sample_ratio * subset)
                end_index = int(len(class_files) * sample_ratio * (subset + 1))

                # Create the subset of files
                sample_files = class_files[start_index:end_index]

                for file in tqdm(
                    sample_files, desc=f"Copying files for class - {class_dir}"
                ):
                    shutil.copy(
                        f"{source_directory}/{class_dir}/{file}",
                        f"{subset_target_directory}/{class_dir}/{file}",
                    )
            else:  # 'fake' directory contains subdirectories
                subdirs = os.listdir(f"{source_directory}/{class_dir}")

                for subdir in subdirs:
                    # Create a directory for each subdirectory of 'fake' in the subset directory
                    os.makedirs(
                        f"{subset_target_directory}/{class_dir}/{subdir}", exist_ok=True
                    )

                    # Obtain a list of files in the current subdirectory
                    subdir_files = os.listdir(
                        f"{source_directory}/{class_dir}/{subdir}"
                    )

                    # Randomly shuffle the file list
                    np.random.shuffle(subdir_files)

                    # Calculate the start and end indices for the current subset
                    start_index = int(len(subdir_files) * sample_ratio * subset)
                    end_index = int(len(subdir_files) * sample_ratio * (subset + 1))

                    # Create the subset of files
                    sample_files = subdir_files[start_index:end_index]

                    for file in tqdm(
                        sample_files,
                        desc=f"Copying files for class - {class_dir}/{subdir}",
                    ):
                        shutil.copy(
                            f"{source_directory}/{class_dir}/{subdir}/{file}",
                            f"{subset_target_directory}/{class_dir}/{subdir}/{file}",
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create stratified subsets from a dataset according to a specified sample ratio. "
        "The dataset is expected to be divided into two classes: 'real' and 'fake'. "
        "The 'fake' class may contain multiple subdirectories. "
        "Each subset is stored in a separate subdirectory named '{source_directory_basename}-{i}' in the target directory."
    )

    parser.add_argument(
        "source_directory", help="The directory of the original dataset."
    )
    parser.add_argument(
        "target_directory", help="The directory where the subsets will be created."
    )
    parser.add_argument(
        "sample_ratio",
        type=float,
        help="The ratio of the original dataset to include in each subset (e.g., 0.1 for 10%%).",
    )

    args = parser.parse_args()

    create_stratified_sample(
        args.source_directory, args.target_directory, args.sample_ratio
    )
