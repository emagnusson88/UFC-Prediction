# UFC Prediction
Using past MMA fight and fighter data to predict the outcome of future bouts

Deployment: https://ufc-prediction.herokuapp.com/

![alt text](https://www.logo-designer.co/wp-content/uploads/2015/07/UFC-logo-design-2015-brand-identity.png)

## Project Structure
```
ufc-prediction
│   readme.md
│   scrape_data.py  -Scrapes event, fight, and fighter data from http://ufcstats.com/statistics/events/completed
│   Initial Load, Cleaning, and Feature Engineering.ipynb  -shows cleaning and transformation process
│   Modeling and Generalizing for New Bouts.ipynb  -shows model iteration and final output
│
└───scraper
│   │   make_soup.py
│   │   print_progress.py 
│   │   scrape_fight_links.py
│   │   scrape_fight_data.py
│   │   scrape_fighter_details.py
│   │   scrape_data.bat -scheduled scraping task to run every Monday at midnight
│   │   Run: scrape_data.py
│
└───data
│   │   total_fight_data.csv -historical UFC fight data (e.g. fight participants, weight class, significant strikes)
│   │   fighter_details.csv -fighter-specific information (e.g. Height, Weight, DOB)
│   │   upcoming_fight_data.csv -future UFC fight data (e.g. fight participants, weight class, location, date)
│
└───cleaning
│   │   Initial Load, Cleaning, and Feature Engineering.ipynb
│   │   Run: clean_data.py
│  
└───modeling
│   │   Modeling and Generalizing for New Bouts.ipynb
│   │   Best so far: SVM classifier (Accuracy=0.78, Precision=0.76, Recall=0.96, AUC=0.70)
│   │   Run: predict.py
│
└───production
    │   Run: streamlit_deploy.py -online deployment for model results and visualizations
    │   URL: https://ufc-prediction.herokuapp.com/

