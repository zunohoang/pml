from pathlib import Path

import pandas as pd


def read_dataset_table(dataset_dir):
    dataset_path = Path(dataset_dir) / "dataset.csv"
    return pd.read_csv(dataset_path)


def remove_ignored_codes(dataset_table, ignored_codes):
    stress_codes = dataset_table["predominant_stress"]
    ignored_rows = stress_codes.isin(ignored_codes)
    kept_rows = ~ignored_rows
    filtered_table = dataset_table.loc[kept_rows]
    return filtered_table.copy()


def assign_labels(dataset_table, label_by_code):
    stress_codes = dataset_table["predominant_stress"]
    labels = stress_codes.map(label_by_code)
    missing_labels = labels.isna()
    if missing_labels.any():
        unmapped_codes = stress_codes.loc[missing_labels]
        unique_codes = unmapped_codes.unique()
        raise ValueError(f"No label configured for predominant_stress codes: {unique_codes}")
    dataset_table["label"] = labels


def add_image_paths(dataset_table, dataset_dir):
    image_directory = Path(dataset_dir) / "images"
    image_paths = []
    for image_id in dataset_table["id"]:
        image_filename = f"{image_id}.jpg"
        image_path = image_directory / image_filename
        image_paths.append(image_path)
    dataset_table["image_path"] = image_paths


def remove_missing_images(dataset_table):
    existing_images = []
    for image_path in dataset_table["image_path"]:
        image_exists = image_path.is_file()
        existing_images.append(image_exists)
    return dataset_table.loc[existing_images]


def build_image_table(config):
    dataset_dir = config["data_root"]
    dataset_table = read_dataset_table(dataset_dir)
    ignored_codes = config["ignored_codes"]
    filtered_table = remove_ignored_codes(dataset_table, ignored_codes)
    label_by_code = config["label_by_code"]
    assign_labels(filtered_table, label_by_code)
    add_image_paths(filtered_table, dataset_dir)
    existing_table = remove_missing_images(filtered_table)
    image_columns = ["id", "label", "image_path"]
    return existing_table[image_columns]
