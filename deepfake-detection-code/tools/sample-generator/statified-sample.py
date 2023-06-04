import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm


def create_stratified_sample(source_directory, target_directory, sample_ratio):
    """
    The function creates a stratified sample from a given dataset directory. It does this by copying a certain
    ratio of data from each class in the source directory to a new target directory. The dataset is assumed to
    be structured such that each class has its own directory, with the 'real' class having its images directly
    under it and the 'fake' class having subdirectories for its images.
    """
    # If the target directory doesn't exist, create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Iterate over the directories in the source directory
    for class_dir in os.listdir(source_directory):
        # Create corresponding directory in target_directory
        os.makedirs(f"{target_directory}/{class_dir}", exist_ok=True)

        # If the current directory is 'real', directly copy images based on the sample_ratio
        if class_dir == "real":
            # Get all files in the 'real' directory
            class_files = os.listdir(f"{source_directory}/{class_dir}")
            # Randomly shuffle the list of files
            np.random.shuffle(class_files)

            # Calculate sample size based on provided ratio
            sample_size = round(len(class_files) * sample_ratio)
            # Get a subset of files for the sample
            sample_files = class_files[:sample_size]

            # Copy the sampled files to the target directory
            for file in tqdm(
                sample_files, desc=f"Copying files for class - {class_dir}"
            ):
                shutil.copy(
                    f"{source_directory}/{class_dir}/{file}",
                    f"{target_directory}/{class_dir}/{file}",
                )
        else:  # If the current directory is 'fake', it has subdirectories
            # Get all subdirectories in 'fake' directory
            subdirs = os.listdir(f"{source_directory}/{class_dir}")

            # Iterate over each subdirectory and copy images based on the sample_ratio
            for subdir in subdirs:
                # Create corresponding subdirectory in target_directory
                os.makedirs(f"{target_directory}/{class_dir}/{subdir}", exist_ok=True)

                # Get all files in the current subdirectory
                subdir_files = os.listdir(f"{source_directory}/{class_dir}/{subdir}")

                # Randomly shuffle the list of files
                np.random.shuffle(subdir_files)

                # Calculate sample size based on provided ratio
                sample_size = round(len(subdir_files) * sample_ratio)
                # Get a subset of files for the sample
                sample_files = subdir_files[:sample_size]

                # Copy the sampled files to the target directory
                for file in tqdm(
                    sample_files, desc=f"Copying files for class - {class_dir}/{subdir}"
                ):
                    shutil.copy(
                        f"{source_directory}/{class_dir}/{subdir}/{file}",
                        f"{target_directory}/{class_dir}/{subdir}/{file}",
                    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a stratified sample from a dataset."
    )
    parser.add_argument(
        "source_directory", help="The directory of the original dataset."
    )
    parser.add_argument(
        "target_directory", help="The directory where the sample will be created."
    )
    parser.add_argument(
        "sample_ratio",
        type=float,
        help="The ratio of the original dataset to include in the sample (e.g., 0.1 for 10%).",
    )

    args = parser.parse_args()

    # Call the function to create the stratified sample
    create_stratified_sample(
        args.source_directory, args.target_directory, args.sample_ratio
    )
