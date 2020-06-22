from createdata.scrape_fight_links import get_all_links
from createdata.scrape_fight_links import get_upcoming_links
from createdata.scrape_fight_links import get_all_new_links
from createdata.scrape_fight_data import create_fight_data_csv
from createdata.scrape_fight_data import create_upcoming_fight_data_csv
from createdata.scrape_fighter_details import create_fighter_data_csv
from datetime import datetime

startTime = datetime.now()

#event_and_fight_links = get_all_links()

##################### uncomment
#event_and_fight_links = get_all_new_links()

#create_fight_data_csv(event_and_fight_links)

create_fighter_data_csv()

#upcoming_event_and_fight_links = get_upcoming_links()

#create_upcoming_fight_data_csv(upcoming_event_and_fight_links)

print('Runtime (min): ', (datetime.now() - startTime).seconds / 60)
