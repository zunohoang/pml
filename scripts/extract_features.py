from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from joblib import Parallel, cpu_count, delayed, parallel_config
from threadpoolctl import threadpool_limits

from coffee_leaf.config import load_config
from coffee_leaf.data.loader import build_image_table
from coffee_leaf.data.split import split_train_test
from coffee_leaf.features.extractor import (
    EXPECTED_FEATURE_COUNT,
    extract_features_from_path,
    list_feature_names,
    validate_feature_vector,
)
from coffee_leaf.features.feature_names import describe_feature


CONFIG_PATH = Path("configs/default.yaml")
FEATURE_FILE_PATH = Path("data/processed/features.npz")
PROGRESS_INTERVAL = 100
SAMPLE_ARRAY_NAMES = ("features", "label_ids", "image_ids")


def load_samples(config):
    image_table = build_image_table(config)
    train_table, _ = split_train_test(image_table, config)
    train_ids = train_table["id"]
    samples = image_table.copy()
    samples["is_train"] = samples["id"].isin(train_ids)
    return samples


def get_worker_count(config):
    if "n_jobs" in config:
        return config["n_jobs"]
    return max(cpu_count() - 1, 1)


def compute_feature_row(sample, max_side):
    image_id = sample.id
    try:
        with threadpool_limits(limits=1):
            feature_vector = extract_features_from_path(sample.image_path, max_side)
        validate_feature_vector(feature_vector)
    except Exception as error:
        return image_id, None, f"{type(error).__name__}: {error}"
    return image_id, feature_vector, None


def collect_feature_rows(results, sample_count):
    feature_rows = []
    image_errors = []
    for completed_count, (image_id, feature_vector, error_message) in enumerate(results, start=1):
        if error_message is None:
            feature_rows.append(feature_vector)
        else:
            image_errors.append((image_id, error_message))
        if completed_count % PROGRESS_INTERVAL == 0:
            print(f"Processed {completed_count}/{sample_count} images", flush=True)
    return feature_rows, image_errors


def compute_all_features(samples, max_side, worker_count):
    sample_rows = samples.itertuples(index=False)
    jobs = (delayed(compute_feature_row)(sample, max_side) for sample in sample_rows)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=worker_count, return_as="generator") as parallel:
            results = parallel(jobs)
            return collect_feature_rows(results, len(samples))


def encode_label_ids(sample_labels, label_names):
    label_ids_by_name = {label_name: label_id for label_id, label_name in enumerate(label_names)}
    encoded_labels = sample_labels.map(label_ids_by_name)
    if encoded_labels.isna().any():
        raise ValueError("Each sample label must appear in label_names.")
    return encoded_labels.to_numpy(dtype=np.int64)


def build_feature_metadata():
    feature_names = list_feature_names()
    feature_regions = []
    feature_groups = []
    for feature_name in feature_names:
        region, group = describe_feature(feature_name)
        feature_regions.append(region)
        feature_groups.append(group)
    return {
        "feature_names": np.asarray(feature_names, dtype=str),
        "feature_regions": np.asarray(feature_regions, dtype=str),
        "feature_groups": np.asarray(feature_groups, dtype=str),
    }


def build_feature_archive(samples, feature_rows, label_names):
    features = np.asarray(feature_rows, dtype=np.float64)
    expected_shape = (len(samples), EXPECTED_FEATURE_COUNT)
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise ValueError(f"Features must be finite and have shape {expected_shape}.")
    archive = build_feature_metadata()
    archive["features"] = features
    archive["label_ids"] = encode_label_ids(samples["label"], label_names)
    archive["label_names"] = np.asarray(label_names, dtype=str)
    archive["image_ids"] = np.asarray(samples["id"].tolist())
    archive["is_train"] = samples["is_train"].to_numpy(dtype=bool)
    return archive


def split_feature_arrays(features_bundle):
    train_mask = features_bundle["is_train"]
    if train_mask.ndim != 1 or train_mask.dtype != bool:
        raise ValueError("is_train must be a one-dimensional boolean array.")
    test_mask = ~train_mask
    train_bundle = {}
    test_bundle = {}
    for array_name, values in features_bundle.items():
        if array_name == "is_train":
            continue
        if array_name in SAMPLE_ARRAY_NAMES:
            if values.ndim == 0 or values.shape[0] != train_mask.size:
                raise ValueError(f"{array_name} must have one row per is_train value.")
            train_bundle[array_name] = values[train_mask]
            test_bundle[array_name] = values[test_mask]
        else:
            train_bundle[array_name] = values
            test_bundle[array_name] = values
    return train_bundle, test_bundle


def write_feature_archives(output_directory, named_archives):
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, _ in named_archives:
        destination = output_directory / filename
        if destination.is_dir():
            raise IsADirectoryError(f"Cannot replace directory with an archive: {destination}")
    with TemporaryDirectory(dir=output_directory) as staging_directory:
        staging_path = Path(staging_directory)
        for filename, archive in named_archives:
            with (staging_path / filename).open("wb") as archive_file:
                np.savez_compressed(archive_file, **archive)
        for filename, _ in named_archives:
            staged_archive = staging_path / filename
            staged_archive.replace(output_directory / filename)


def save_feature_file(output_path, samples, feature_rows, label_names):
    archive = build_feature_archive(samples, feature_rows, label_names)
    output_path = Path(output_path)
    named_archives = ((output_path.name, archive),)
    write_feature_archives(output_path.parent, named_archives)


def save_split_feature_files(output_path, features_bundle):
    train_bundle, test_bundle = split_feature_arrays(features_bundle)
    output_path = Path(output_path)
    train_filename = f"{output_path.stem}_train.npz"
    test_filename = f"{output_path.stem}_test.npz"
    named_archives = ((train_filename, train_bundle), (test_filename, test_bundle))
    write_feature_archives(output_path.parent, named_archives)
    for filename, archive in named_archives:
        sample_count = archive["image_ids"].size
        print(f"Saved {sample_count} images to {output_path.parent / filename}")


def print_image_errors(image_errors):
    print(f"Feature extraction failed for {len(image_errors)} image(s):")
    for image_id, error_message in image_errors:
        print(f"Image {image_id}: {error_message}")


def main(config_path=CONFIG_PATH, output_path=FEATURE_FILE_PATH):
    config = load_config(config_path)
    samples = load_samples(config)
    worker_count = get_worker_count(config)
    feature_rows, image_errors = compute_all_features(samples, config["image_max_size"], worker_count)
    if image_errors:
        print_image_errors(image_errors)
        return 1
    label_by_code = config["label_by_code"]
    label_names = [label_by_code[code] for code in sorted(label_by_code)]
    features_bundle = build_feature_archive(samples, feature_rows, label_names)
    save_split_feature_files(output_path, features_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
