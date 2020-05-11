from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests

def make_soup(url: str) -> BeautifulSoup:
  source_code = requests.get(url, allow_redirects=False)
  plain_text = source_code.text.encode('ascii', 'replace')
  return BeautifulSoup(plain_text,'html.parser')
