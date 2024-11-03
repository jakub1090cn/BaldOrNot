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


def get_image_urls(api_key, cse_id, query, num_images):
    service = build("customsearch", "v1", developerKey=api_key)
    image_urls = []

    for start in range(1, num_images + 1, 10):
        res = (
            service.cse()
            .list(
                q=query,
                cx=cse_id,
                searchType="image",
                num=min(10, num_images - len(image_urls)),
                start=start,
            )
            .execute()
        )

        if "items" not in res:
            break

        for item in res["items"]:
            image_urls.append(item["link"])

    return image_urls


def detect_faces(image, min_face_area_ratio=0.1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.02, minNeighbors=5
    )

    img_height, img_width = gray_image.shape
    img_area = img_height * img_width

    valid_faces = []

    for x, y, w, h in faces:
        face_area = w * h
        face_area_ratio = face_area / img_area

        if face_area_ratio >= min_face_area_ratio:
            valid_faces.append((x, y, w, h))

    return valid_faces


def download_images(image_urls, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for i, url in enumerate(image_urls):
        try:
            print(f"Downloading {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            image_data = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if image is not None:
                faces = detect_faces(image)

                if len(faces) == 1:
                    image_filename = os.path.join(
                        download_path, f"image_{i + 1}.jpg"
                    )
                    cv2.imwrite(image_filename, image)
                    print(f"Saved image with 1 face: {image_filename}")
                else:
                    print(f"Skipped image (found {len(faces)} faces): {url}")
            else:
                print(f"Could not decode image from {url}")

        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    config = CLI(GoogleApiConfig)

    api_key = config.google_api_data.api_key
    cse_id = config.google_api_data.cse_id

    query = config.download_params.query
    download_path = config.download_params.download_path
    num_images = config.download_params.num_images

    face_cascade = load_face_cascade()

    print(f"Searching for '{query}' on Google...")
    image_urls = get_image_urls(api_key, cse_id, query, num_images)
    print(f"Found {len(image_urls)} images.")
    download_images(image_urls, download_path)
    print("Download complete.")
