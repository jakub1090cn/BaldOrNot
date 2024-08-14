import pandas as pd
from src.data import check_sample_images, prepare_merged_dataframe


def test_check_sample_images(monkeypatch):
    """
    Tests the `check_sample_images` function to ensure that it correctly identifies empty or corrupted files in a
    specified directory.

    The test verifies that the `check_sample_images` function:
    - Accurately identifies corrupted or empty files.
    - Correctly counts the number of valid, successfully loaded images.

    Asserts:
        empty_or_corrupted == ['corrupted.txt']: Confirms that the function correctly identifies the corrupted file
        (simulated as `corrupted.txt`).
        num_correct == 1: Confirms that the function correctly counts one valid image in the directory.
    """
    directory = "test_images"

    empty_or_corrupted, num_correct = check_sample_images(directory)

    assert empty_or_corrupted == ["corrupted.txt"]
    assert num_correct == 1


def test_prepare_merged_dataframe(mocker):
    """
    Tests the `prepare_merged_dataframe` function, which merges two CSV files
    into a single DataFrame based on a common `image_id` column.

    This test:
    1. Mocks the `open` function to simulate opening CSV files without actually
       requiring the files to exist on disk.
    2. Uses `mocker.patch` to replace `pandas.read_csv` so that it reads from
       the mocked file objects instead of actual files.
    3. Defines expected data in a DataFrame that represents the correct merging
       of the two input CSV files.
    4. Compares the DataFrame returned by `prepare_merged_dataframe` to the expected
       DataFrame using `pd.testing.assert_frame_equal`.

    The test passes if the merged DataFrame matches the expected DataFrame, indicating
    that the function correctly reads and merges the CSV files.
    """
    mock_subsets = mocker.mock_open(
        read_data="image_id,subset\n0001.jpg,1\n0002.jpg,2"
    )
    mocker.patch("builtins.open", mock_subsets, create=True)

    mocker.patch(
        "pandas.read_csv",
        side_effect=[
            pd.read_csv(mock_subsets()),
            pd.read_csv(
                mocker.mock_open(
                    read_data="image_id,label\n0001.jpg,0\n0002.jpg,1"
                )()
            ),
        ],
    )

    subsets_path = "fake_subsets.csv"
    labels_path = "fake_labels.csv"

    expected_df = pd.DataFrame(
        {
            "image_id": ["0001.jpg", "0002.jpg"],
            "subset": [1, 2],
            "label": [0, 1],
        }
    )

    result_df = prepare_merged_dataframe(subsets_path, labels_path)

    pd.testing.assert_frame_equal(result_df, expected_df)
