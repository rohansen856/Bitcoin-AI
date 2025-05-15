import os
import platform
import subprocess
import traceback
from html.parser import HTMLParser
from io import StringIO
import xml.etree.ElementTree as ET
import py7zr

import requests
from loguru import logger
from tqdm import tqdm


def download_dump(download_path):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    logger.info('Downloading the data from archive...')
    archive_url = "https://archive.org/download/stackexchange/bitcoin.stackexchange.com.7z"
    try:
        r = requests.get(archive_url, stream=True)
        if r.status_code == 200:
            with open(download_path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=1024)):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Successfully downloaded data to path: {download_path}")
        else:
            logger.error(f"Request returned an error: {r.status_code}")
    except requests.RequestException as e:
        logger.error(f"An error occurred while downloading: {e}")


def extract_dump(download_path, extract_path):
    try:
        logger.info('Extracting the data using py7zr...')
        with py7zr.SevenZipFile(download_path, mode='r') as archive:
            archive.extractall(path=extract_path)
        logger.info(f"Extraction successful to path: {os.path.abspath(extract_path)}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")

def parse_users(users_file_path) -> dict:
    users = {}
    tree = ET.parse(users_file_path)
    root = tree.getroot()
    for user in root:
        users[user.attrib.get("Id")] = user.attrib.get("DisplayName")
    logger.info(f"Number of users found: {len(users.keys())}")
    return users


def parse_posts(posts_file_path):
    tree = ET.parse(posts_file_path)
    root = tree.getroot()
    return root


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
