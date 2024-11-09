import os
import requests
import cv2
import numpy as np
from googleapiclient.discovery import build
from jsonargparse import CLI
import logging
from datetime import datetime


from src.config_class import GoogleApiConfig
from src.constants import SCRAPPING_LOG_FILE_NAME
from src.logging import setup_logging


def load_face_cascade() -> cv2.CascadeClassifier:
    """Loads Haar Cascade classifier to detect face."""
    face_cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cv2.CascadeClassifier(face_cascade_path)


def get_image_urls(config: GoogleApiConfig):
    """
    Retrieves image URLs based on a search query using Google Custom Search API.

    Args:
        config (GoogleApiConfig): Configuration object containing API key, CSE ID,
                                  search parameters, and the desired number of images.

    Returns:
        list: A list of image URLs obtained from the search.
    """
    # Initialize the Google Custom Search API service
    service = build(
        "customsearch", "v1", developerKey=config.google_api_data.api_key
    )

    # Prepare variables for the query
    image_urls = []
    query = config.search_params.query
    cse_id = config.google_api_data.cse_id
    num_images = config.search_params.num_images
    batch_size = (
        10  # maximum number of results per page in a single API request
    )

    # Iterate in batches to retrieve the desired number of images
    for start in range(1, num_images + 1, batch_size):
        # Perform a search request with specified parameters
        response = (
            service.cse()
            .list(
                q=query,
                cx=cse_id,
                searchType="image",
                num=min(
                    batch_size, num_images - len(image_urls)
                ),  # Get up to batch_size images or remaining count
                start=start,
            )
            .execute()
        )

        # Check if the response contains image results
        if "items" not in response:
            break  # Exit loop if there are no more results

        # Add each image URL to the list
        image_urls.extend(item["link"] for item in response["items"])

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


def download_image(url: str) -> np.ndarray | None:
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
        logging.error(f"Failed to download {url}: {e}")
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
    download_path = os.path.join(
        config.download_params.scrapper_output_path, "images"
    )
    os.makedirs(download_path, exist_ok=True)

    if len(faces) == 1:
        image_filename = os.path.join(download_path, f"image_{index + 1}.jpg")
        cv2.imwrite(image_filename, image)
        logging.info(f"Saved image with 1 face: {image_filename}")
    else:
        logging.info(f"Skipped image (found {len(faces)} faces).")


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
        logging.info(f"Downloading {url}")
        image = download_image(url)

        if image is not None:
            faces = detect_faces(image, config, face_cascade)
            save_image_with_face(image, faces, i, config)
        else:
            logging.info(f"Could not decode image from {url}")


if __name__ == "__main__":
    config = CLI(GoogleApiConfig)

    output_dir_path = os.path.join(
        config.download_params.scrapper_output_path,
        f"web_scrapper_logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(output_dir_path, exist_ok=True)

    setup_logging(output_dir_path, SCRAPPING_LOG_FILE_NAME)

    face_cascade = load_face_cascade()

    logging.info(f"Searching for '{config.search_params.query}' on Google...")
    image_urls = get_image_urls(config)
    logging.info(f"Found {len(image_urls)} images.")
    process_images(image_urls, config, face_cascade)
    logging.info("Download complete.")
