import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config import BaldOrNotConfig
from src.data import BaldDataset
from src.constants import (
    N_CHANNELS_RGB,
    N_CHANNELS_GRAYSCALE,
    DEFAULT_IMG_SIZE,
)


@pytest.fixture
def sample_df():
    data = {
        "image_id": ["bald.jpg", "not_bald.jpg"],
        "labels": [1, 0],
        "partition": [0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def config():
    return BaldOrNotConfig()


def test_init(sample_df):
    dataset = BaldDataset(sample_df)
    assert dataset.batch_size == 32
    assert dataset.dim == DEFAULT_IMG_SIZE
    assert dataset.n_channels == N_CHANNELS_RGB
    assert dataset.shuffle is True
    assert len(dataset.indexes) == len(sample_df)


@pytest.mark.parametrize(
    "n_channels, expected", [(N_CHANNELS_RGB, 3), (N_CHANNELS_GRAYSCALE, 1)]
)
def test_n_channels_accepts_valid_values(sample_df, n_channels, expected):
    dataset = BaldDataset(sample_df, n_channels=n_channels)
    assert dataset.n_channels == expected, f"n_channels should be {expected}."


def test_n_channels_rejects_invalid_value(sample_df):
    # Test with an invalid n_channels value
    with pytest.raises(
        ValueError,
        match="n_channels must be either 1 \("  # noqa: W605
        "grayscale\) or 3 \(RGB\)\.",  # noqa: W605
    ):
        BaldDataset(sample_df, n_channels=2)


@pytest.mark.parametrize(
    "batch_size, expected_length",
    [
        (1, 2),  # With batch size 1, there should be 2 batches
        (2, 1),  # With batch size 2, there should be 1 batch
    ],
)
def test_len(sample_df, batch_size, expected_length):
    dataset = BaldDataset(sample_df, batch_size=batch_size)
    assert len(dataset) == expected_length


def test_getitem_calculates_indices_correctly(sample_df):
    dataset = BaldDataset(sample_df, batch_size=1, shuffle=False)

    # Test the first batch
    expected_indices = [0]  # Indices for the first batch
    actual_indices = dataset.indexes[0:1]
    assert (
        list(actual_indices) == expected_indices
    ), "Indices for the first batch are incorrect."

    # Test the second batch
    expected_indices = [1]  # Indices for the second batch
    actual_indices = dataset.indexes[1:2]
    assert (
        list(actual_indices) == expected_indices
    ), "Indices for the second batch are incorrect."


def test_getitem_extracts_image_ids_correctly(sample_df):
    dataset = BaldDataset(sample_df, batch_size=1, shuffle=False)

    # Test the first batch
    expected_list_IDs_temp = ["bald.jpg"]
    actual_list_IDs_temp = [dataset.list_IDs[i] for i in dataset.indexes[0:1]]
    assert (
        actual_list_IDs_temp == expected_list_IDs_temp
    ), "Image IDs for the first batch are incorrect."

    # Test the second batch
    expected_list_IDs_temp = ["not_bald.jpg"]
    actual_list_IDs_temp = [dataset.list_IDs[i] for i in dataset.indexes[1:2]]
    assert (
        actual_list_IDs_temp == expected_list_IDs_temp
    ), "Image IDs for the second batch are incorrect."


@patch.object(BaldDataset, "_BaldDataset__data_preprocessing")
def test_getitem_calls_data_preprocessing_correctly(
    mock_data_preprocessing, sample_df
):
    mock_data_preprocessing.return_value = (
        np.zeros((2, *DEFAULT_IMG_SIZE, 3)),  # Mocked X (images)
        np.array([1, 0]),  # Mocked y (labels)
    )

    dataset = BaldDataset(sample_df, batch_size=2, shuffle=False)

    # Simulate the first batch to check the call to __data_preprocessing
    dataset[0]
    expected_list_IDs_temp = ["bald.jpg", "not_bald.jpg"]
    mock_data_preprocessing.assert_called_with(expected_list_IDs_temp)

    # There is no second batch, so no need to simulate it


@patch.object(BaldDataset, "_BaldDataset__data_preprocessing")
def test_getitem_returns_correct_X_and_y(mock_data_preprocessing, sample_df):
    # Mock the return value of __data_preprocessing
    mock_data_preprocessing.return_value = (
        np.array([[[[0.1]], [[0.2]], [[0.3]]]]),  # Mocked X (images)
        np.array([1, 0]),  # Mocked y (labels)
    )

    dataset = BaldDataset(sample_df, batch_size=2, shuffle=False)

    # Test the first (and only) batch
    X, y = dataset[0]
    expected_X = np.array([[[[0.1]], [[0.2]], [[0.3]]]])
    expected_y = np.array([1, 0])
    assert np.array_equal(
        X, expected_X
    ), "Returned X (images) for the first batch is incorrect."
    assert np.array_equal(
        y, expected_y
    ), "Returned y (labels) for the first batch is incorrect."

    # There is no second batch, so no need to test it


@pytest.mark.parametrize("shuffle", [True, False])
def test_on_epoch_end(sample_df, shuffle):
    dataset = BaldDataset(sample_df, shuffle=shuffle)
    initial_indexes = dataset.indexes.copy()
    dataset.on_epoch_end()

    assert len(dataset.indexes) == len(initial_indexes)

    if shuffle:
        assert not np.array_equal(dataset.indexes, initial_indexes)
    else:
        assert np.array_equal(dataset.indexes, initial_indexes)


def test_data_preprocessing_initializes_matrices_correctly(
    sample_df, config
):  # problem
    batch_size = 2

    with patch.object(config.paths, "images_dir", new="src/samples"):
        with patch("src.data.BaldOrNotConfig", return_value=config):  # E501
            dataset = BaldDataset(sample_df, batch_size=batch_size)
            X, y = dataset._BaldDataset__data_preprocessing(
                ["bald.jpg", "not_bald.jpg"]
            )

            assert X.shape == (
                batch_size,
                *DEFAULT_IMG_SIZE,
                N_CHANNELS_RGB,
                "X matrix has incorrect shape.",
            )
            assert y.shape == (batch_size,), "y matrix has incorrect shape."
            assert y.dtype == int, "y should contain integers."


def test_get_wrong_files_list():
    with patch("os.listdir", return_value=["bald.jpg", "not_bald.jpg"]):
        with patch(
            "cv2.imread", side_effect=[None, np.ones((300, 300, 3))]
        ):  # noqa: E501
            wrong_files = BaldDataset._BaldDataset__get_wrong_files_list(
                "src/samples"
            )
            assert len(wrong_files) == 1
            assert wrong_files[0] == "bald.jpg"


def test_get_cleaned_df(sample_df):
    with patch(
        "src.data.BaldDataset._BaldDataset__get_wrong_files_list",
        return_value=["not_bald.jpg"],
    ):
        cleaned_df = BaldDataset.get_cleaned_df(
            sample_df, images_dir="src/samples"
        )
        assert len(cleaned_df) == 1
        assert "not_bald.jpg" not in cleaned_df["image_id"].values


def test_prepare_merged_dataframe(tmpdir):
    subsets_path = tmpdir.join("subsets.csv")
    labels_path = tmpdir.join("labels.csv")

    subsets_df = pd.DataFrame(
        {"image_id": ["bald.jpg", "not_bald.jpg"], "partition": [0, 1]}
    )
    labels_df = pd.DataFrame(
        {"image_id": ["bald.jpg", "not_bald.jpg"], "Bald": [1, 0]}
    )

    subsets_df.to_csv(subsets_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    df_merged = BaldDataset.prepare_merged_dataframe(subsets_path, labels_path)

    assert isinstance(df_merged, pd.DataFrame), (
        "The result should be a " "DataFrame."
    )
    assert list(df_merged.columns) == ["image_id", "partition", "labels"], (
        "The DataFrame should contain only the columns: 'image_id', "
        "'partition', and 'labels'."
    )
    assert len(df_merged) == 2, "The DataFrame should contain 2 rows."


def test_create_subset_dfs(sample_df):
    extra_data = pd.DataFrame(
        {
            "image_id": ["new_sample.jpg"],
            "labels": [1],
            "partition": [0],  # New sample since we need three for this test
        }
    )

    extended_sample_df = pd.concat([sample_df, extra_data], ignore_index=True)

    train_df, val_df, test_df = BaldDataset.create_subset_dfs(
        extended_sample_df
    )
    assert len(train_df) == 1
    assert len(val_df) == 1
    assert len(test_df) == 1
    assert "partition" not in test_df.columns
