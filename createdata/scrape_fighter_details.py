import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from pathlib import Path
import pickle
import os
from createdata.print_progress import print_progress
from createdata.make_soup import make_soup
from typing import List, Dict, Tuple

HEADER = ['Height', 'Weight', 'Reach', 'Stance', 'DOB']
BASE_PATH = Path(os.getcwd())/'data'
CSV_PATH = BASE_PATH/'fighter_details.csv'
PAST_FIGHTER_LINKS_PATH = BASE_PATH/'fighter_links.pickle'

def get_fighter_group_urls() -> List[str]:
	alphas = [chr(i) for i in range(ord('a'),ord('a')+26)]
	fighter_group_urls = [f"http://ufcstats.com/statistics/fighters?char={alpha}&page=all" for alpha in alphas]

	return fighter_group_urls

def get_fighter_name_and_link(fighter_group_urls: List[str]) -> Dict[str, List[str]]:
	fighter_name_and_link = {}
	fighter_name = ''

	l = len(fighter_group_urls)
	print('Scraping all fighter names and links: ')
	print_progress(0, l, prefix = 'Progress:', suffix = 'Complete')

	for index, fighter_group_url in enumerate(fighter_group_urls):
	    soup = make_soup(fighter_group_url)
	    table = soup.find('tbody')
	    names = table.findAll('a', {'class': 'b-link b-link_style_black'}, href=True)
	    for i, name in enumerate(names):
	        if (i+1)%3 != 0:
	            if fighter_name == '':
	                fighter_name = name.text
	            else:
	                fighter_name = fighter_name + ' ' + name.text
	        else:
	            fighter_name_and_link[fighter_name] = name['href']
	            fighter_name = ''
	    print_progress(index + 1, l, prefix = 'Progress:', suffix = 'Complete')


	pickle_in = open(PAST_FIGHTER_LINKS_PATH.as_posix(),"rb")

	past_fighter_links = pickle.load(pickle_in)

	new_fighter_links = list(fighter_name_and_link.values())

	fighter_links = np.setdiff1d(new_fighter_links,past_fighter_links)

	pickle_in.close()

	#set event links to the newly scraped list
	pickle_out = open(PAST_FIGHTER_LINKS_PATH.as_posix(),"wb")
	pickle.dump(new_fighter_links, pickle_out)
	pickle_out.close()

	fighter_name_and_link = dict(filter(lambda elem: elem[1] not in fighter_links,fighter_name_and_link.items()))

	return fighter_name_and_link

def get_fighter_name_and_details(fighter_name_and_link: Dict[str, List[str]]) -> Dict[str, List[str]]:
	fighter_name_and_details = {}

	l = len(fighter_name_and_link)
	print('Scraping all fighter data: ')
	print_progress(0, l, prefix = 'Progress:', suffix = 'Complete')

	for index, (fighter_name, fighter_url) in enumerate(fighter_name_and_link.items()):
	    another_soup = make_soup(fighter_url)
	    divs = another_soup.findAll('li', {'class':"b-list__box-list-item b-list__box-list-item_type_block"})
	    data = []
	    for i, div in enumerate(divs):
	        if i == 5:
	            break
	        data.append(div.text.replace('  ', '').replace('\n', '').replace('Height:', '').replace('Weight:', '')\
	                   .replace('Reach:', '').replace('STANCE:', '').replace('DOB:', ''))

	    fighter_name_and_details[fighter_name] = data
	    print_progress(index + 1, l, prefix = 'Progress:', suffix = 'Complete')

	return fighter_name_and_details

def create_fighter_data_csv() -> None:
	fighter_group_urls = get_fighter_group_urls()
	fighter_name_and_details = get_fighter_name_and_details(get_fighter_name_and_link(fighter_group_urls))

	df = pd.DataFrame(fighter_name_and_details).T.replace('--', value=np.NaN).replace('', value=np.NaN)
	df.columns = HEADER

	existing_data = pd.read_csv(CSV_PATH.as_posix())

	final_df = pd.concat([df,existing_data]).drop_duplicates()

	final_df.to_csv(CSV_PATH.as_posix(), index_label = 'fighter_name')
