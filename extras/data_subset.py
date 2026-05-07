import os
import random
import shutil
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===
anchor_res = "9km"  # Only 9km as anchor
_pass = 'backward' # forward, backward, complete

# spatial_split_id = 1  # Spatial split ID

# Base input directories
src_root = Path("Dataset")
src_dirs = {
    "1km": src_root / "1km",
    "9km": src_root / "9km"
}

# Destination directories
dst_base_dirs = {
    "train": Path(f"training/temporal/{_pass}/train"),
    "test": Path(f"training/temporal/{_pass}/test")
}

# dst_base_dirs = {
#     "train": Path(f"training/spatial/{spatial_split_id}/train"),
#     "test": Path(f"training/spatial/{spatial_split_id}/test")
# }

# Create destination folders
for base in dst_base_dirs.values():
    # for res in ["1km", "3km", "9km"]:
    for res in ["1km", "9km"]:
        os.makedirs(base / res, exist_ok=True)

# === Load anchor files ===
files_anchor = sorted([f for f in os.listdir(src_dirs[anchor_res]) if f.endswith(".tif")])

# Utility to generalize resolution in filename
def get_stem(filename):
    parts = filename.replace(".tif", "").split("_")
    parts[1] = "<RES>"
    return "_".join(parts) + ".tif"

# Collect valid doubles, high res and low res
valid_triplets = []

for file_anchor in files_anchor:
    stem = get_stem(file_anchor)

    # extract prefix from 9km
    prefix = file_anchor.split("_")[0]


    file_1km = stem.replace("<RES>", "1km")
    # file_3km = stem.replace("<RES>", "3km")
    file_9km = file_anchor

    path_1km = src_dirs["1km"] / file_1km
    # path_3km = src_dirs["3km"] / file_3km
    path_9km = src_dirs["9km"] / file_9km

    # Require all files to exist
    if path_1km.exists() and path_9km.exists():
        try:
            # Parse date from anchor filename
            date_str = file_anchor.split("_")[3]
            date_obj = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception as e:
            print(f"Failed to parse date from: {file_anchor} — {e}")
            continue

        valid_triplets.append({
            "1km": file_1km,
            # "3km": file_3km,
            "9km": file_9km,
            "date": date_obj
        })

def split_bycount(valid_doubles, split_ratio=0.8, total_samples=100, seed=42):
    """
    Splits the valid_doubles into train and test sets based on ratio and sample count.

    Parameters:
        valid_doubles (list): List of double dictionaries.
        split_ratio (float): Proportion of samples to use for training (e.g., 0.8).
        total_samples (int): Total number of doubles to select from the full list.
        seed (int): Random seed for reproducibility.

    Returns:
        train_set (list): Subset of doubles for training.
        test_set (list): Subset of doubles  for testing.
    """
    assert len(valid_doubles) >= total_samples, \
        f"Only found {len(valid_doubles)} valid doubles. Needed: {total_samples}"

    random.seed(seed)
    random.shuffle(valid_doubles)

    selected = valid_doubles[:total_samples]
    split_index = int(split_ratio * total_samples)

    train_set = selected[:split_index]
    test_set = selected[split_index:]

    return train_set, test_set


def split_bydate(valid_doubles, trainY = [2015, 2015], testY= [2023]):
    """
    Splits the valid_doubles into train and test sets based on ratio and sample count.

    Parameters:
        valid_doubles (list): List of double dictionaries.


    Returns:
        train_set (list): Subset of doubles for training.
        test_set (list): Subset of doubles for testing.
    """
    # Extract the data for train set
    try:
        train_start, train_end = trainY
        train_set = [double for double in valid_doubles if train_start <= double["date"].year <= train_end]
    except:
        train_y = trainY
        train_set = [double for double in valid_doubles if double["date"].year == train_y[0]]

    try:
        test_start, test_end = testY
        test_set = [double for double in valid_doubles if test_start <= double["date"].year <= test_end]
    except:
        test_y = testY
        test_set = [double for double in valid_doubles if double["date"].year == test_y[0]]

    # # Split into train/test based on year
    # train_set = [triplet for triplet in valid_triplets if 2014 <= triplet["date"].year <= 2018]
    # test_set = [triplet for triplet in valid_triplets if triplet["date"].year == 2019]

    # Check that we have enough
    assert len(train_set) >= 1, "Not enough training."
    assert len(test_set) >= 1, "Not enough test data."
    assert len(train_set) >= len(test_set), "train data less then test data."

    return train_set, test_set


def copy_set(doubles, dst_dir):
    for double in doubles:
        # for res in ["1km", "3km", "9km"]:
        for res in ["1km", "9km"]:
            src_path = src_dirs[res] / double[res]
            dst_path = dst_dir / res / double[res]
            shutil.copy2(src_path, dst_path)


def split_byid(valid_doubles, user_ids):
    """
    Splits the valid_doubles into train and test sets based on user-defined IDs.

    Parameters:
        valid_doubles (list): List of double dictionaries.
        user_ids (list): List of IDs to use for splitting.

    Returns:
        train_set (list): Subset of doubles for training (not in user-defined IDs).
        test_set (list): Subset of doubles corresponding to user-defined IDs.
    """
    if not user_ids:
        raise ValueError("You must provide a list of user-defined IDs.")

    # Extract test set based on user-defined IDs
    test_set = [
        double for double in valid_doubles
        if any(str(user_id) == double["9km"].split("_")[-1].replace(".tif", "") for user_id in user_ids)
    ]

    # Compute the train set (remaining data not in the test set)
    test_files = set([double["9km"] for double in test_set])  # Collect test file names
    train_set = [double for double in valid_doubles if double["9km"] not in test_files]

    # Check that we have enough data
    assert len(train_set) >= 1, "Not enough training data."
    assert len(test_set) >= 1, "Not enough test data."

    return train_set, test_set


def copy_resampled(source_dir, copy_from_dir, target_dir):
    # Create target directory if doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    missing_files = []

    # Iterate through files in the source directory
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        
        # Check if it's a file and not a directory
        if os.path.isfile(source_path):
            copy_path = os.path.join(copy_from_dir, file_name)
            target_path = os.path.join(target_dir, file_name)

            # Check if file exists in the copying directory
            if os.path.exists(copy_path):
                # Copy the file to the target directory
                shutil.copy2(copy_path, target_path)
                # print(f"Copied: {file_name} -> {target_dir}")
            else:
                print(f"Missing: {file_name}")
                if file_name.lower().endswith(".tif"):
                    missing_files.append(file_name)

    # Report missing files
    if missing_files:
        print("\nThe following .tif files are missing in the copying directory:")
        for missing_file in missing_files:
            print(missing_file)


def copy_lst(source_dir, copy_from_dir, target_dir):
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    missing_files = []

    # Iterate through the files in the source directory
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)

        # Check if it's a file (and not a directory)
        if os.path.isfile(source_path):
            # Extract the reference part from the source file name
            # Example: For "SMAP-E_1km_AM_20150403_0.tif", extract "AM_20150403_0"
            parts = file_name.split("_")
            if len(parts) >= 4:
                time_part = "_".join(parts[2:])  # Extract time-related parts like AM_20150403_0
                lst_file_name = f"MYD11A1-LST_1km_{time_part}"

                # Construct paths
                copy_path = os.path.join(copy_from_dir, lst_file_name)
                target_path = os.path.join(target_dir, lst_file_name)

                # Check if the corresponding LST file exists in the copy_from_dir
                if os.path.exists(copy_path):
                    # Copy the file to the target directory
                    shutil.copy2(copy_path, target_path)
                    # print(f"Copied: {lst_file_name} -> {target_dir}")
                else:
                    print(f"Missing: {lst_file_name}")
                    if lst_file_name.lower().endswith(".tif"):
                        missing_files.append(lst_file_name)

    # Report missing files
    if missing_files:
        print("\nThe following .tif files are missing in the copying directory:")
        for missing_file in missing_files:
            print(missing_file)


# === Main Processing ===

# # # Example Usage:
# user_ids = [spatial_split_id]  # Define the user IDs for splitting

# # # Split data into train and test sets by user-defined IDs
# train_set, test_set = split_byid(valid_triplets, user_ids)


# train_set, test_set = split_bycount(valid_triplets, split_ratio=0.8, total_samples=5146)

if _pass == 'forward':
    # forwards
    train_set, test_set = split_bydate(valid_triplets, trainY = [2015, 2021], testY= [2022, 2024])

# backwards
if _pass == 'backward':
    train_set, test_set = split_bydate(valid_triplets, trainY = [2017, 2024], testY= [2015, 2016])

# Full
if _pass == 'complete':
    train_set, test_set = split_bydate(valid_triplets, trainY = [2015, 2024], testY= [2023, 2024])

# Step 6: Copy files
copy_set(train_set, dst_base_dirs["train"])
copy_set(test_set, dst_base_dirs["test"])

print(f"Copied {len(train_set)} training triplets and {len(test_set)} testing triplets.")

# Copy resampled 1km-r-m files AFTER above !!
# Example usage
for path in ['test', 'train']:

    source_directory = f"training/temporal/{_pass}/{path}/1km"
    copy_from_directory = "Dataset/1km-r-m"
    target_directory = f"training/temporal/{_pass}/{path}/1km-r-m"

    copy_resampled(source_directory, copy_from_directory, target_directory)

