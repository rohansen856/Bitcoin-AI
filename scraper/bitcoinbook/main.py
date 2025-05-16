import json
import re
import requests
from datetime import datetime
from loguru import logger
import re

def extract_level_2_headings(text):
    # Match lines that start with exactly two '=' followed by a space
    pattern = re.compile(r'^==\s+(.*)', re.MULTILINE)
    return pattern.findall(text)

site = 'https://uithub.com/bitcoinbook/bitcoinbook/blob/develop'
chapters = ['/ch01_intro.adoc', '/ch02_overview.adoc', '/ch03_bitcoin-core.adoc', '/ch04_keys.adoc',
            '/ch05_wallets.adoc', '/ch06_transactions.adoc', '/ch07_authorization-authentication.adoc', '/ch08_signatures.adoc',
            '/ch09_fees.adoc', '/ch10_network.adoc', '/ch11_blockchain.adoc', '/ch12_mining.adoc', '/ch13_security.adoc' , '/ch14_applications.adoc']
chapter_links = [f"{site}{chapter}" for chapter in chapters]

documents = []


if __name__ == "__main__":

    for url in chapter_links:
        data = requests.get(url).text
        
        title = extract_level_2_headings(data)[0]
        body = data.strip()
        body_type = "md"
        author = "Andreas Antonopoulous"
        chapter_number = ''.join(re.findall(r'\d+', url))
        id = "bitcoinbook-chapter-" + chapter_number
        tags = ""
        domain = "https://github.com/bitcoinbook/bitcoinbook"
        url = url
        created_at = "2022-11-15"  # date of most recent commit

        document = {
            "title": title,
            "body": body,
            "body_type": body_type,
            "author": author,
            "id": id,
            "tags": tags,
            "domain": domain,
            "url": url,
            "created_at": created_at,
            "indexed_at": datetime.utcnow().isoformat()
        }

        logger.info(document.get("id"))
        documents.append(document)
    print("Number of documents: " + str(len(documents)))

    with open("./data/bitcoinbook.json", "w") as f:
        json.dump(documents, f, indent=4)

    # Close the file
    f.close()
