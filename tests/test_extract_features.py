from pathlib import Path

import numpy as np
import pandas as pd

from scripts import extract_features


def create_samples():
    return pd.DataFrame({
        "id": [27, 10, 14, 99, 36],
        "label": ["rust", "healthy", "leaf_miner", "cercospora", "brown_spot"],
        "image_path": [Path(f"{image_id}.jpg") for image_id in (27, 10, 14, 99, 36)],
        "is_train": [True, False, True, True, False],
    })


def create_fake_vectors(samples):
    return [np.full(403, image_id, dtype=np.float32) for image_id in samples["id"]]


def test_save_feature_file_round_trip(tmp_path):
    samples = create_samples()
    feature_rows = create_fake_vectors(samples)
    label_names = ["healthy", "leaf_miner", "rust", "brown_spot", "cercospora"]
    output_path = tmp_path / "processed" / "features.npz"
    extract_features.save_feature_file(output_path, samples, feature_rows, label_names)
    expected_archive = extract_features.build_feature_archive(samples, feature_rows, label_names)
    with np.load(output_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "features", "label_ids", "label_names", "image_ids", "is_train",
            "feature_names", "feature_regions", "feature_groups",
        }
        assert archive["features"].shape == (5, 403)
        assert archive["features"].dtype == np.float64
        assert archive["is_train"].dtype == bool
        np.testing.assert_array_equal(archive["image_ids"], [27, 10, 14, 99, 36])
        np.testing.assert_array_equal(archive["label_ids"], [2, 0, 1, 4, 3])
        for key, expected_values in expected_archive.items():
            np.testing.assert_array_equal(archive[key], expected_values)


def test_saving_twice_writes_identical_bytes(tmp_path):
    samples = create_samples()
    feature_rows = create_fake_vectors(samples)
    label_names = ["healthy", "leaf_miner", "rust", "brown_spot", "cercospora"]
    output_path = tmp_path / "features.npz"
    extract_features.save_feature_file(output_path, samples, feature_rows, label_names)
    first_file_bytes = output_path.read_bytes()
    extract_features.save_feature_file(output_path, samples, feature_rows, label_names)
    assert output_path.read_bytes() == first_file_bytes


def test_split_feature_arrays_preserves_sample_order_and_metadata():
    samples = create_samples()
    feature_rows = create_fake_vectors(samples)
    label_names = ["healthy", "leaf_miner", "rust", "brown_spot", "cercospora"]
    bundle = extract_features.build_feature_archive(samples, feature_rows, label_names)
    train_bundle, test_bundle = extract_features.split_feature_arrays(bundle)
    expected_keys = set(bundle) - {"is_train"}
    assert set(train_bundle) == set(test_bundle) == expected_keys
    for split_bundle, expected_rows in ((train_bundle, [0, 2, 3]), (test_bundle, [1, 4])):
        assert split_bundle["features"].shape == (len(expected_rows), 403)
        for array_name in ("features", "label_ids", "image_ids"):
            np.testing.assert_array_equal(split_bundle[array_name], bundle[array_name][expected_rows])
        for array_name in ("label_names", "feature_names", "feature_regions", "feature_groups"):
            np.testing.assert_array_equal(split_bundle[array_name], bundle[array_name])


def fake_extract_features_from_path(image_path, max_side):
    image_id = int(image_path.stem)
    if image_id in (14, 36):
        raise ValueError(f"broken image {image_id}")
    return np.full(403, image_id, dtype=np.float64)


def test_compute_all_features_collects_errors_and_keeps_input_order(monkeypatch):
    samples = create_samples()
    monkeypatch.setattr(extract_features, "extract_features_from_path", fake_extract_features_from_path)
    feature_rows, image_errors = extract_features.compute_all_features(samples, 256, 1)
    feature_ids = [vector[0] for vector in feature_rows]
    assert feature_ids == [27, 10, 99]
    assert image_errors == [(14, "ValueError: broken image 14"), (36, "ValueError: broken image 36")]


def load_fake_config(config_path):
    return {"image_max_size": 256, "n_jobs": 1}


def load_fake_samples(config):
    return create_samples()


def test_main_reports_image_errors_and_does_not_write_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(extract_features, "load_config", load_fake_config)
    monkeypatch.setattr(extract_features, "load_samples", load_fake_samples)
    monkeypatch.setattr(extract_features, "extract_features_from_path", fake_extract_features_from_path)
    output_path = tmp_path / "processed" / "features.npz"
    exit_code = extract_features.main(output_path=output_path)
    captured_output = capsys.readouterr().out
    assert exit_code == 1
    assert "Image 14: ValueError: broken image 14" in captured_output
    assert "Image 36: ValueError: broken image 36" in captured_output
    assert not output_path.exists()
    assert not output_path.with_name("features_train.npz").exists()
    assert not output_path.with_name("features_test.npz").exists()
