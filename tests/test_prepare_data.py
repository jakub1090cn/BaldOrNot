from src.data import check_sample_images, create_data_subsets, prepare_merged_dataframe


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
    directory = 'test_images'

    empty_or_corrupted, num_correct = check_sample_images(directory)

    assert empty_or_corrupted == ['corrupted.txt']
    assert num_correct == 1


