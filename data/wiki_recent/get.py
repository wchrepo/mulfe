import requests
from bs4 import BeautifulSoup
import os

def get_recent_pages(limit=10):
    """
    Fetches the titles of the most recently created Wikipedia pages.
    """
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "list": "recentchanges",
        "rctype": "new",
        "rcnamespace": 0,
        "rclimit": limit,
        "format": "json"
    }

    response = S.get(url=URL, params=PARAMS)
    data = response.json()

    pages = []
    for item in data['query']['recentchanges']:
        pages.append(item['title'])

    return pages

def get_first_paragraph(page_title):
    """
    Fetches the first paragraph of a Wikipedia page given its title.
    """
    S = requests.Session()
    URL = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"

    response = S.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the first paragraph
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        if paragraph.text.strip() != "":
            return paragraph.text.strip()

    return None

def main():
    recent_pages = get_recent_pages(10)
    for page in recent_pages:
        first_paragraph = get_first_paragraph(page)
        if first_paragraph:
            print(f"Page: {page}\nFirst Paragraph: {first_paragraph}\n")

if __name__ == "__main__":
    os.environ['http_proxy'] = 'http://127.0.0.1:56666'
    os.environ['https_proxy'] = 'http://127.0.0.1:56666'
    main()