import requests
from bs4 import BeautifulSoup
import pickle
import os
from pathlib import Path
from urllib.request import urlopen
from typing import List, Dict, Tuple
from createdata.make_soup import make_soup
import numpy as np

UPCOMING_ODDS_URL = 'https://www.oddsshark.com/ufc/odds'
UPCOMING_EVENTS_URL = 'http://ufcstats.com/statistics/events/upcoming?page=all'
BASE_PATH = Path(os.getcwd())/'data'
ODDS_PATH = BASE_PATH/'event_and_fight_links.pickle'

def get_all_odds(odds_url: str=UPCOMING_ODDS_URL) -> List[str]:
	links = []
	url = all_events_url
	soup = make_soup(UPCOMING_ODDS_URL)
	for link in soup.findAll('div',{'class': 'op-content-wrapper'}):
		for href in link.findAll('a'):
			foo = href.get('href')
			links.append(foo)


	return links
