import os
from jsonargparse import CLI
from datetime import datetime

import logging
from src.config_class import GoogleApiConfig
from src.constants import SCRAPPING_LOG_FILE_NAME
from src.logging import setup_logging
from src.web_scrapper import load_face_cascade, get_image_urls, process_images


def run_scrapper(config: GoogleApiConfig):
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


if __name__ == "__main__":
    config = CLI(GoogleApiConfig)
    run_scrapper(config)
