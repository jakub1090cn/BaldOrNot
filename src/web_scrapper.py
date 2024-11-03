import os
import requests
import cv2
import numpy as np
from googleapiclient.discovery import build
from jsonargparse import CLI

from src.config_class import GoogleApiConfig


def load_face_cascade() -> cv2.CascadeClassifier:
    """Loads Haar Cascade classifier to detect face."""
    face_cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cv2.CascadeClassifier(face_cascade_path)


def get_image_urls(config: GoogleApiConfig):
    service = build(
        "customsearch", "v1", developerKey=config.google_api_data.api_key
    )
    image_urls = []

    for start in range(1, config.search_params.num_images + 1, 10):
        res = (
            service.cse()
            .list(
                q=config.search_params.query,
                cx=config.google_api_data.cse_id,
                searchType="image",
                num=min(10, config.search_params.num_images - len(image_urls)),
                start=start,
            )
            .execute()
        )

        if "items" not in res:
            break

        for item in res["items"]:
            image_urls.append(item["link"])

    return image_urls


def detect_faces(
    image, config: GoogleApiConfig, face_cascade: cv2.CascadeClassifier
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=config.face_detection_params.scale_factor,
        minNeighbors=config.face_detection_params.min_neighbors,
    )

    img_height, img_width = gray_image.shape
    img_area = img_height * img_width

    valid_faces = []

    for x, y, w, h in faces:
        face_area = w * h
        face_area_ratio = face_area / img_area

        if face_area_ratio >= config.face_detection_params.min_face_area_ratio:
            valid_faces.append((x, y, w, h))

    return valid_faces


def download_image(url: str) -> np.ndarray:
    """Downloads an image from the given URL.

    Args:
        url (str): URL of the image to download.

    Returns:
        np.ndarray: Image in NumPy format or None if download failed.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None


def save_image_with_face(
    image: np.ndarray, faces: list, index: int, config: GoogleApiConfig
):
    """Saves the image if exactly one face is detected.

    Args:
        image (np.ndarray): Image to save.
        faces (list): List of detected faces.
        index (int): Image index, used in the filename.
        config (GoogleApiConfig): Configuration with the download path.
    """
    download_path = config.download_params.download_path
    os.makedirs(download_path, exist_ok=True)

    if len(faces) == 1:
        image_filename = os.path.join(download_path, f"image_{index + 1}.jpg")
        cv2.imwrite(image_filename, image)
        print(f"Saved image with 1 face: {image_filename}")
    else:
        print(f"Skipped image (found {len(faces)} faces).")


def process_images(
    image_urls: list,
    config: GoogleApiConfig,
    face_cascade: cv2.CascadeClassifier,
):
    """Downloads images from URLs, detects faces, and saves images with exactly one face.

    Args:
        image_urls (list): List of image URLs.
        config (GoogleApiConfig): Configuration containing download and face detection parameters.
        face_cascade (cv2.CascadeClassifier): Haar classifier for face detection.
    """
    for i, url in enumerate(image_urls):
        print(f"Downloading {url}")
        image = download_image(url)

        if image is not None:
            faces = detect_faces(image, config, face_cascade)
            save_image_with_face(image, faces, i, config)
        else:
            print(f"Could not decode image from {url}")


if __name__ == "__main__":
    config = CLI(GoogleApiConfig)
    face_cascade = load_face_cascade()

    print(f"Searching for '{config.search_params.query}' on Google...")
    image_urls = get_image_urls(config)
    print(f"Found {len(image_urls)} images.")
    process_images(image_urls, config, face_cascade)
    print("Download complete.")
