import os
import shutil
from loguru import logger
from utils import download_dump, extract_dump

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == "__main__":
    DOWNLOAD_PATH = os.path.join(DATA_DIR, "bitcoin.stackexchange.com.7z")
    EXTRACT_PATH = os.path.join(DATA_DIR, "bitcoin.stackexchange.com")
    ZIP_OUTPUT_PATH = os.path.join(DATA_DIR, "bitcoin_stackexchange.zip")

    # Step 1: Download .7z if not already downloaded
    if not os.path.exists(DOWNLOAD_PATH):
        logger.info("Downloading Bitcoin StackExchange dump...")
        download_dump(DOWNLOAD_PATH)
        logger.success(f'Downloaded file to: {os.path.abspath(DOWNLOAD_PATH)}')
    else:
        logger.info(f'File already exists at path: {os.path.abspath(DOWNLOAD_PATH)}')

    # Step 2: Extract if not already extracted
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)
        should_extract = True
    else:
        if not os.listdir(EXTRACT_PATH):
            should_extract = True
        else:
            file_count = len(os.listdir(EXTRACT_PATH))
            logger.info(f'{file_count} files already exist at path: {os.path.abspath(EXTRACT_PATH)}')
            should_extract = False

    if should_extract:
        logger.info("Extracting the dump...")
        extract_dump(DOWNLOAD_PATH, EXTRACT_PATH)
        logger.success("Extraction complete.")

    # Step 3: Zip the extracted folder
    if not os.path.exists(ZIP_OUTPUT_PATH):
        logger.info("Creating ZIP archive of extracted data...")
        shutil.make_archive(
            base_name=os.path.splitext(ZIP_OUTPUT_PATH)[0],
            format='zip',
            root_dir=EXTRACT_PATH
        )
        logger.success(f"ZIP archive created at: {ZIP_OUTPUT_PATH}")
    else:
        logger.info(f"ZIP file already exists at: {ZIP_OUTPUT_PATH}")

    logger.info("Data download, extraction, and zipping complete.")
