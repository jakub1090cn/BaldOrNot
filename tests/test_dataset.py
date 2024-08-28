import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config import BaldOrNotConfig
from src.data import BaldDataset


@pytest.fixture
def sample_df():
    data = {
        "image_id": ["bald.jpg", "not_bald.jpg", "bald_or_not.jpg"],
        "labels": [1, 0, 1],
        "partition": [0, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def config_mock():
    return {"paths.images_dir": "src/samples"}


def test_init(sample_df):
    dataset = BaldDataset(sample_df)
    assert dataset.batch_size == 32
    assert dataset.dim == (218, 178)
    assert dataset.n_channels == 3
    assert dataset.n_classes == 2
    assert dataset.shuffle is True
    assert len(dataset.indexes) == len(sample_df)


@pytest.mark.parametrize(
    "batch_size, expected_length",
    [
        (1, 3),  # With batch size 1, there should be 3 batches
        (2, 1),  # With batch size 2, there should be 1 batch
        (3, 1),  # With batch size 3, there should be 1 batch
    ],
)
def test_len(sample_df, batch_size, expected_length):
    dataset = BaldDataset(sample_df, batch_size=batch_size)
    assert len(dataset) == expected_length


def test_getitem_calculates_indices_correctly(sample_df):
    dataset = BaldDataset(sample_df, batch_size=2, shuffle=False)

    # Test the first batch
    expected_indices = [0, 1]  # Indices for the first batch
    actual_indices = dataset.indexes[0:2]
    assert (
        list(actual_indices) == expected_indices
    ), "Indices for the first batch are incorrect."
    expected_indices = [2]  # Index for the second batch
    actual_indices = dataset.indexes[2:3]
    assert (
        list(actual_indices) == expected_indices
    ), "Indices for the second batch are incorrect."


def test_getitem_extracts_image_ids_correctly(sample_df):
    dataset = BaldDataset(sample_df, batch_size=2, shuffle=False)

    # Test the first batch
    expected_list_IDs_temp = ["bald.jpg", "not_bald.jpg"]
    actual_list_IDs_temp = [dataset.list_IDs[i] for i in dataset.indexes[0:2]]
    assert (
        actual_list_IDs_temp == expected_list_IDs_temp
    ), "Image IDs for the first batch are incorrect."

    # Test the second batch
    expected_list_IDs_temp = ["bald_or_not.jpg"]
    actual_list_IDs_temp = [dataset.list_IDs[i] for i in dataset.indexes[2:3]]
    assert (
        actual_list_IDs_temp == expected_list_IDs_temp
    ), "Image ID for the second batch is incorrect."


@patch.object(BaldDataset, "_BaldDataset__data_generation")
def test_getitem_calls_data_generation_correctly(
    mock_data_generation, sample_df
):
    mock_data_generation.return_value = (
        np.zeros((2, 218, 178, 3)),  # Mocked X (images)
        np.array([1, 0]),  # Mocked y (labels)
    )

    dataset = BaldDataset(sample_df, batch_size=2, shuffle=False)

    # Simulate the first batch to check the call to __data_generation
    dataset[0]
    expected_list_IDs_temp = ["bald.jpg", "not_bald.jpg"]
    mock_data_generation.assert_called_with(expected_list_IDs_temp)

    # Change the values for the second batch
    mock_data_generation.return_value = (
        np.zeros((1, 218, 178, 3)),  # Mocked X (images) for second batch
        np.array([1]),  # Mocked y (labels) for second batch
    )

    # Simulate the second batch to check the call to __data_generation
    dataset[1]
    expected_list_IDs_temp = ["bald_or_not.jpg"]
    mock_data_generation.assert_called_with(expected_list_IDs_temp)


@patch.object(BaldDataset, "_BaldDataset__data_generation")
def test_getitem_returns_correct_X_and_y(mock_data_generation, sample_df):
    # Mock the return value of __data_generation
    mock_data_generation.return_value = (
        np.array([[[[0.1]], [[0.2]], [[0.3]]]]),  # Mocked X (images)
        np.array([1, 0, 1]),  # Mocked y (labels)
    )

    dataset = BaldDataset(sample_df, batch_size=2, shuffle=False)

    # Test the first batch
    X, y = dataset[0]
    expected_X = np.array([[[[0.1]], [[0.2]], [[0.3]]]])
    expected_y = np.array([1, 0, 1])
    assert np.array_equal(
        X, expected_X
    ), "Returned X (images) for the first batch is incorrect."
    assert np.array_equal(
        y, expected_y
    ), "Returned y (labels) for the first batch is incorrect."

    # Test the second batch
    X, y = dataset[1]
    assert np.array_equal(
        X, expected_X
    ), "Returned X (images) for the second batch is incorrect."
    assert np.array_equal(
        y, expected_y
    ), "Returned y (labels) for the second batch is incorrect."


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


def test_data_generation_initializes_matrices_correctly(sample_df):  # problem
    config_instance = BaldOrNotConfig()

    with patch.object(config_instance.paths, "images_dir", new="src/samples"):
        with patch(
            "src.data.BaldOrNotConfig", return_value=config_instance
        ):  # E501
            dataset = BaldDataset(sample_df, batch_size=2)
            X, y = dataset._BaldDataset__data_generation(
                ["bald.jpg", "not_bald.jpg"]
            )

            assert X.shape == (2, 178, 218, 3), "X matrix has incorrect shape."
            assert y.shape == (2,), "y matrix has incorrect shape."
            assert y.dtype == int, "y should contain integers."


# here two more __data_generation tests: for imgs transforming and result


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
        assert len(cleaned_df) == 2
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
    train_df, val_df, test_df = BaldDataset.create_subset_dfs(sample_df)
    assert len(train_df) == 1
    assert len(val_df) == 1
    assert len(test_df) == 1
    assert "partition" not in test_df.columns
