import os
import cv2
from typing import Tuple, List


def check_images(directory: str) -> Tuple[List[str], int, int]:
    empty_or_corrupted: List[str] = []
    num_correct: int = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            img = cv2.imread(file_path)

            if img is None or img.size == 0:
                empty_or_corrupted.append(filename)
            else:
                num_correct += 1
        except Exception:
            empty_or_corrupted.append(filename)

    return empty_or_corrupted, len(empty_or_corrupted), num_correct
