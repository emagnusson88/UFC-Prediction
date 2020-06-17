import requests
from bs4 import BeautifulSoup
import pickle
import os
from pathlib import Path
from urllib.request import urlopen
from typing import List, Dict, Tuple
from createdata.make_soup import make_soup
import numpy as np

ALL_EVENTS_URL = 'http://ufcstats.com/statistics/events/completed?page=all'
UPCOMING_EVENTS_URL = 'http://ufcstats.com/statistics/events/upcoming?page=all'
BASE_PATH = Path(os.getcwd())/'data'
EVENT_AND_FIGHT_LINKS_PATH = BASE_PATH/'event_and_fight_links.pickle'
PAST_EVENT_LINKS_PATH = BASE_PATH/'past_event_links.pickle'

def get_link_of_past_events(all_events_url: str=ALL_EVENTS_URL) -> List[str]:
	links = []
	url = all_events_url
	soup = make_soup(all_events_url)
	for link in soup.findAll('td',{'class': 'b-statistics__table-col'}):
		for href in link.findAll('a'):
			foo = href.get('href')
			links.append(foo)
	pickle_out = open(PAST_EVENT_LINKS_PATH.as_posix(),"wb")
	pickle.dump(links, pickle_out)
	pickle_out.close()

	return links

def get_event_and_fight_links(event_links: List[str]) -> Dict[str, List[str]]:
	event_and_fight_links = {}
	for link in event_links:
		event_fights = []
		soup = make_soup(link)
		for row in soup.findAll('tr', {'class': 'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'}):
			href = row.get('data-link')
			event_fights.append(href)
		event_and_fight_links[link] = event_fights

	pickle_out = open(EVENT_AND_FIGHT_LINKS_PATH.as_posix(),"wb")
	pickle.dump(event_and_fight_links, pickle_out)
	pickle_out.close()

	return event_and_fight_links

def get_all_links() -> Dict[str, List[str]]:
	if EVENT_AND_FIGHT_LINKS_PATH.exists()!=True:
		if PAST_EVENT_LINKS_PATH.exists()!=True:
			past_event_links = get_link_of_past_events()
		else:
			pickle_in = open(PAST_EVENT_LINKS_PATH.as_posix(),"rb")
			past_event_links = pickle.load(pickle_in)
			pickle_in.close()
		event_and_fight_links = get_event_and_fight_links(past_event_links)
	else:
		pickle_in = open(EVENT_AND_FIGHT_LINKS_PATH.as_posix(),"rb")
		event_and_fight_links = pickle.load(pickle_in)
		pickle_in.close()

	return event_and_fight_links
################################
def get_link_of_past_events_no_pickle(all_events_url: str=ALL_EVENTS_URL) -> List[str]:
    links = []
    url = all_events_url
    soup = make_soup(all_events_url)
    for link in soup.findAll('td',{'class': 'b-statistics__table-col'}):
        for href in link.findAll('a'):
            foo = href.get('href')
            links.append(foo)

    return links

def get_link_of_upcoming_events_no_pickle(upcoming_events_url: str=UPCOMING_EVENTS_URL) -> List[str]:
    links = []
    url = upcoming_events_url
    soup = make_soup(upcoming_events_url)
    for link in soup.findAll('td',{'class': 'b-statistics__table-col'}):
        for href in link.findAll('a'):
            foo = href.get('href')
            links.append(foo)

    return links

def get_event_and_fight_links_no_pickle(event_links: List[str]) -> Dict[str, List[str]]:
	event_and_fight_links = {}
	for link in event_links:
		event_fights = []
		soup = make_soup(link)
		for row in soup.findAll('tr', {'class': 'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'}):
			href = row.get('data-link')
			event_fights.append(href)
		event_and_fight_links[link] = event_fights

	return event_and_fight_links

def get_all_new_links() -> Dict[str, List[str]]:
	if EVENT_AND_FIGHT_LINKS_PATH.exists()!=True:
		if PAST_EVENT_LINKS_PATH.exists()!=True:
			past_event_links = get_link_of_past_events()
		else:
			pickle_in = open(PAST_EVENT_LINKS_PATH.as_posix(),"rb")
			past_event_links = pickle.load(pickle_in)
			pickle_in.close()
		event_and_fight_links = get_event_and_fight_links(past_event_links)
	else:

		pickle_in = open(PAST_EVENT_LINKS_PATH.as_posix(),"rb")
		past_event_links = pickle.load(pickle_in)

		new_event_links = get_link_of_past_events_no_pickle()

		event_and_fight_links = get_event_and_fight_links_no_pickle(np.setdiff1d(new_event_links,past_event_links))

		pickle_in.close()

		#set event links to the newly scraped list
		pickle_out = open(PAST_EVENT_LINKS_PATH.as_posix(),"wb")
		pickle.dump(new_event_links, pickle_out)
		pickle_out.close()

	return event_and_fight_links

def get_upcoming_links() -> Dict[str, List[str]]:

	upcoming_event_links = get_link_of_upcoming_events_no_pickle()
	upcoming_event_and_fight_links = get_event_and_fight_links_no_pickle(upcoming_event_links)

	return upcoming_event_and_fight_links

###############################3333

def get_this_event_fight_links(event_links: List[str]) -> Dict[str, List[str]]:
	if isinstance(event_links, list):
		return get_event_and_fight_links(event_links)
	else:
		return get_event_and_fight_links(list[event_links])
