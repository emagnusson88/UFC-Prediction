#!/usr/bin/env python
# coding: utf-8

# Executable version of cleaning jupyter notebook

# <h1>After Scraping: Cleaning and Feature Engineering</h1>
#
# - Acknowledgements:
#     - ufcstats for comprehensive data sets on past MMA bouts: http://ufcstats.com/
#     - Rajeev Warrier for providing the groundwork for the web scraper: https://github.com/WarrierRajeev/UFC-Predictions

# In[240]:


import pandas as pd
import numpy as np

DATA_PATH ='./data'
df_fighters = pd.read_csv(DATA_PATH+'/fighter_details.csv')
df_fights = pd.read_csv(DATA_PATH+'/total_fight_data.csv', sep=';')


# In[241]:


df_fighters.head(3)


# In[242]:


df_fights.head(3)


# <h3>Processing Fighter data set</h3>

# In[243]:


df_fighters.isna().sum()


# - fighters with NaN Weight values have little to no useful data
#     - therefore, these rows will be excluded

# In[244]:


df_fighters[pd.isnull(df_fighters['Weight'])].isna().sum()


# In[245]:


df_fighters = df_fighters[df_fighters['Weight'].notna()]


# - to fill NaN values in bodily metrics, find:
#     - average reach for each height increment
#     - average height for each weight increment

# In[246]:


df_fighters['Weight'] = df_fighters['Weight'].apply(lambda x: x.split(' ')[0])
df_fighters['Weight'] = df_fighters['Weight'].astype(float)


# In[247]:


df_fighters['Height'] = df_fighters['Height'].fillna('0\' 0\"')
df_fighters['Height'] = df_fighters['Height'].apply(lambda x: int(x.split('\' ')[0])*12 + int(x.split('\' ')[1].replace('\"','')))
df_fighters['Height'] = df_fighters['Height'].replace(0, np.nan).astype(float)


# In[248]:


df_fighters['Height'] = df_fighters.groupby('Weight')['Height'].apply(lambda x: x.fillna(x.mean()))
df_fighters['Height'] = df_fighters['Height'].fillna(df_fighters['Height'].mean())


# In[249]:


df_fighters['Reach'] = df_fighters['Reach'].fillna('0')
df_fighters['Reach'] = df_fighters['Reach'].apply(lambda x: x.replace('\"',''))
df_fighters['Reach'] = df_fighters['Reach'].replace('0', np.nan).astype(float)


# In[250]:


df_fighters['Reach'] = df_fighters.groupby('Height')['Reach'].apply(lambda x: x.fillna(x.mean()))
df_fighters['Reach'] = df_fighters['Reach'].fillna(df_fighters['Reach'].mean())


# In[251]:


df_fighters['Stance'].value_counts()


# <h3>Processing Fight data set</h3>

# - split attack stats into attempts/landed numerical format

# In[252]:


df_fights.columns
attack_cols = ['R_SIG_STR.', 'B_SIG_STR.','R_TOTAL_STR.', 'B_TOTAL_STR.',
       'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY',
       'B_BODY', 'R_LEG', 'B_LEG', 'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH',
       'B_CLINCH', 'R_GROUND', 'B_GROUND']


# In[253]:


for col in attack_cols:
    df_fights[col+'_ATT'] = df_fights[col].apply(lambda x: int(x.split('of')[1]))
    df_fights[col+'_LANDED'] = df_fights[col].apply(lambda x: int(x.split('of')[0]))


# In[254]:


df_fights.drop(attack_cols, axis=1, inplace=True)


# - check for NULL values

# In[255]:


for col in df_fights:
    if df_fights[col].isnull().sum()!=0:
        print(f'Null count in {col} = {df_fights[col].isnull().sum()}')


# In[256]:


df_fights[df_fights['Winner'].isnull()]['win_by'].value_counts()


# In[257]:


df_fights['Winner'].fillna('Draw', inplace=True)


# - convert percentages to decimal values

# In[258]:


percentage_columns = ['R_SIG_STR_pct', 'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']

for col in percentage_columns:
    df_fights[col] = df_fights[col].apply(lambda x : float(x.replace('%',''))/100)


# - isolating Title fights and weight classes

# In[259]:


df_fights['Fight_type'].value_counts()[df_fights['Fight_type'].value_counts() > 1].index


# In[260]:


df_fights['title_bout'] = df_fights['Fight_type'].apply(lambda x: 1 if 'Title Bout' in x else 0)


# In[261]:


weight_classes = ['Women\'s Strawweight', 'Women\'s Bantamweight',
                  'Women\'s Featherweight', 'Women\'s Flyweight', 'Lightweight',
                  'Welterweight', 'Middleweight','Light Heavyweight',
                  'Heavyweight', 'Featherweight','Bantamweight', 'Flyweight', 'Open Weight']

def make_weight_class(x):
    for weight_class in weight_classes:
        if weight_class in x:
            return weight_class
    if x == 'Catch Weight Bout' or 'Catchweight Bout':
        return 'Catch Weight'
    else:
        return 'Open Weight'


# In[262]:


df_fights['weight_class'] = df_fights['Fight_type'].apply(make_weight_class)


# In[263]:


df_fights['weight_class'].value_counts()


# - isolate total fight time (seconds)

# In[264]:


df_fights['Format'].value_counts()


# In[265]:


time_in_first_round = {'3 Rnd (5-5-5)': 5*60,
                       '5 Rnd (5-5-5-5-5)': 5*60,
                       '1 Rnd + OT (12-3)': 12*60,
                       'No Time Limit': 1,
                       '3 Rnd + OT (5-5-5-5)': 5*60,
                       '1 Rnd (20)': 1*20,
                       '2 Rnd (5-5)': 5*60,
                       '1 Rnd (15)': 15*60,
                       '1 Rnd (10)': 10*60,
                       '1 Rnd (12)':12*60,
                       '1 Rnd + OT (30-5)': 30*60,
                       '1 Rnd (18)': 18*60,
                       '1 Rnd + OT (15-3)': 15*60,
                       '1 Rnd (30)': 30*60,
                       '1 Rnd + OT (31-5)': 31*5,
                       '1 Rnd + OT (27-3)': 27*60,
                       '1 Rnd + OT (30-3)': 30*60}

exception_format_time = {'1 Rnd + 2OT (15-3-3)': [15*60, 3*60],
                         '1 Rnd + 2OT (24-3-3)': [24*60, 3*60]}


# In[266]:


# Converting to seconds
df_fights['last_round_time'] = df_fights['last_round_time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))


# In[267]:


def get_total_time(row):
    if row['Format'] in time_in_first_round.keys():
        return (row['last_round'] - 1) * time_in_first_round[row['Format']] + row['last_round_time']
    elif row['Format'] in exception_format_time.keys():
        if (row['last_round'] - 1) >= 2:
            return exception_format_time[row['Format']][0] + (row['last_round'] - 2) *                     exception_format_time[row['Format']][1] + row['last_round_time']
        else:
            return (row['last_round'] - 1) * exception_format_time[row['Format']][0] + row['last_round_time']


# In[268]:


df_fights['total_time_fought(sec)'] = df_fights.apply(get_total_time, axis=1)


# In[269]:


def get_num_rounds(x):
    if x == 'No Time Limit':
        return 1
    else:
        return len((x.split('(')[1].replace(')','').split('-')))

df_fights['no_of_rounds'] = df_fights['Format'].apply(get_num_rounds)


# - there are too many distinct locations
#     - in order to create a more signifcant feature, location is adapted to a binary indicator of whether or not the fight took place in Las Vegas, Nevada (i.e. the most popular fight location)

# In[270]:


df_fights['location'].value_counts()


# In[271]:


df_fights['location']=df_fights['location'].apply(lambda x: 1 if str(x).find('Las Vegas')!=-1 else 0)


# - change Date of Birth and fight date from string to datetime

# In[272]:


from datetime import datetime

df_fighters['DOB']=df_fighters['DOB'].astype(str)

month_code = {'Jan': 'January',
      'Feb': 'February',
      'Mar': 'March',
      'Apr': 'April',
      'May': 'May',
      'Jun': 'June',
      'Jul': 'July',
      'Aug': 'August',
      'Sep': 'September',
      'Oct': 'October',
      'Nov': 'November',
      'Dec': 'December'}

for k, v in month_code.items():
    df_fighters['DOB'] = df_fighters['DOB'].apply(lambda x: x.replace(k, v) if type(x) == str else x)

#df_fighters['DOB'] = df_fighters['DOB'].apply(lambda row: datetime.strptime(row, '%B %d, %Y') if type(row) == str else row)
#df_fights['date'] = df_fights['date'].apply(lambda row: datetime.strptime(row, '%B %d, %Y') if type(row) == str else row)

df_fighters['DOB'] = pd.to_datetime(df_fighters['DOB'])
df_fights['date'] = pd.to_datetime(df_fights['date'])


# - recode winner column to binary and drop obsolete columns

# In[273]:


df_fights['Red_win'] = df_fights.apply(lambda row: 1 if row['Winner'] == row['R_fighter'] else 0, axis=1)

df_fights.drop(columns = ['Format', 'Referee','Fight_type','last_round_time'], inplace=True)


# - recode win_by feature into bins for Submission, KO, or Other

# In[274]:


df_fights['win_by'].value_counts()


# In[275]:


df_fights['win_by'] = df_fights.apply(lambda row: 'Submission' if 'Submission' in row['win_by']
                                                  else('KO' if 'KO' in row['win_by']
                                                  else 'Other'), axis=1)


# <h3>Consolidate red/blue corner stats to align them with the correct fighter</h3>

# In[276]:


df_red = df_fights[['R_fighter','R_KD', 'R_SIG_STR_pct',
       'R_TD_pct', 'R_SUB_ATT',
       'R_PASS', 'R_REV',
       'R_SIG_STR._ATT', 'R_SIG_STR._LANDED',
       'R_TOTAL_STR._ATT',
       'R_TOTAL_STR._LANDED',
       'R_TD_ATT', 'R_TD_LANDED', 'R_HEAD_ATT',
       'R_HEAD_LANDED', 'R_BODY_ATT',
       'R_BODY_LANDED',  'R_LEG_ATT',
       'R_LEG_LANDED',  'R_DISTANCE_ATT',
       'R_DISTANCE_LANDED',
       'R_CLINCH_ATT', 'R_CLINCH_LANDED',
       'R_GROUND_ATT', 'R_GROUND_LANDED',
       'Winner', 'win_by', 'last_round',
       'date', 'location',
       'title_bout', 'weight_class', 'total_time_fought(sec)', 'no_of_rounds']]

df_blue = df_fights[['B_fighter',  'B_KD',
       'B_SIG_STR_pct','B_TD_pct', 'B_SUB_ATT',
       'B_PASS',  'B_REV',
       'B_SIG_STR._ATT', 'B_SIG_STR._LANDED',
       'B_TOTAL_STR._ATT', 'B_TOTAL_STR._LANDED',
       'B_TD_ATT', 'B_TD_LANDED',
       'B_HEAD_ATT', 'B_HEAD_LANDED',
       'B_BODY_ATT', 'B_BODY_LANDED',
       'B_LEG_ATT', 'B_LEG_LANDED',
       'B_DISTANCE_ATT', 'B_DISTANCE_LANDED',
       'B_CLINCH_ATT', 'B_CLINCH_LANDED',
       'B_GROUND_ATT', 'B_GROUND_LANDED',
       'Winner', 'win_by', 'last_round',
       'date', 'location',
       'title_bout', 'weight_class', 'total_time_fought(sec)', 'no_of_rounds']]


# - get rid of red/blue corner prefixes in order to union fighter history

# In[277]:


def drop_prefix(self, prefix):
    self.columns = self.columns.str.replace('^'+prefix,'')
    return self

pd.core.frame.DataFrame.drop_prefix = drop_prefix


# In[278]:


union = pd.concat([df_red.drop_prefix('R_'), df_blue.drop_prefix('B_')])


# - join this combined fight history DataFrame to the originial fighter DataFrame

# In[279]:


union[union['fighter']=='Daniel Cormier'].head(3)


# In[280]:


union.head()


# In[281]:


df_fighters.head()


# In[282]:


df_fighter_history = pd.merge(df_fighters, union, left_on='fighter_name', right_on='fighter', how='left', indicator=True)


# - 1,330 fighters without any fight stats (in original fighter dataset)
#     - However, every fighter involved in a historical bout is contained in the original fighter dataset
#     - UPDATE: after analysis using the above 1,330 fighters, they will be dropped to ensure data quality and avoid "garbage in, garbage out

# In[283]:


df_fighter_history._merge.value_counts()


# In[284]:


df_fighter_history = df_fighter_history[df_fighter_history._merge != 'left_only']


# In[285]:


union.shape


# - replace categorical feature (i.e. Stance) NULLs with the mode of that column

# In[286]:


df_fighter_history


# In[287]:


df_fighter_history['Stance'].fillna(df_fighter_history['Stance'].value_counts().index[0], inplace=True)


# In[288]:


df_fighter_history.shape


# - lack of depth in individual fight history presents a problem for forecasting fighter performance

# In[289]:


df_fighter_history['fighter_name'].value_counts()


# <h3>Feature Engineering</h3>

# In[290]:


df_fights[df_fights['B_fighter']=='Omar Morales']


# In[291]:


df_fighter_history.head()


# - creating age (at fight date) feature

# In[293]:


df_fighter_history['age'] = df_fighter_history['date'] - df_fighter_history['DOB']
df_fighter_history['age']=df_fighter_history['age']/np.timedelta64(1,'Y')
df_fighter_history['age'].fillna(df_fighter_history['age'].mean(), inplace=True)


# In[294]:


df_fighter_history['age'].sort_values()


# In[295]:


df_fighter_history['age'].mean()


# In[296]:


df_fighter_history.drop(columns='_merge', inplace=True)


# - create features for 1) # of fights they've been in, 2) what % they won, and 3) the ranked order of past fights

# In[297]:


df_fighter_history['num_fights'] = df_fighter_history['date'].groupby(df_fighter_history['fighter_name']).transform('count')

df_fighter_history['win'] = df_fighter_history.apply(lambda row: 1 if row['Winner'] == row['fighter_name'] else 0, axis=1)
df_fighter_history.drop(columns=['Winner'], inplace=True)

df_fighter_history['num_wins'] = df_fighter_history['win'].groupby(df_fighter_history['fighter_name']).transform('sum')

df_fighter_history['record'] = df_fighter_history['num_wins']/df_fighter_history['num_fights']


# In[298]:


df_fighter_history['title_bout']=df_fighter_history['title_bout'].apply(lambda x: 1 if x == 1 else 0)


# In[299]:


df_fighter_history['fight_rank']=df_fighter_history.groupby('fighter_name')['date'].rank(ascending=True, method='first')


# In[300]:


df_fighter_history.drop(columns='fighter', inplace=True)


# In[301]:


df_fights_train = df_fights[['R_fighter', 'B_fighter', 'R_KD', 'B_KD', 'R_SIG_STR_pct',
       'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct', 'R_SUB_ATT', 'B_SUB_ATT',
       'R_PASS', 'B_PASS', 'R_REV', 'B_REV', 'win_by', 'last_round', 'date',
       'location', 'R_SIG_STR._ATT', 'R_SIG_STR._LANDED',
       'B_SIG_STR._ATT', 'B_SIG_STR._LANDED', 'R_TOTAL_STR._ATT',
       'R_TOTAL_STR._LANDED', 'B_TOTAL_STR._ATT', 'B_TOTAL_STR._LANDED',
       'R_TD_ATT', 'R_TD_LANDED', 'B_TD_ATT', 'B_TD_LANDED', 'R_HEAD_ATT',
       'R_HEAD_LANDED', 'B_HEAD_ATT', 'B_HEAD_LANDED', 'R_BODY_ATT',
       'R_BODY_LANDED', 'B_BODY_ATT', 'B_BODY_LANDED', 'R_LEG_ATT',
       'R_LEG_LANDED', 'B_LEG_ATT', 'B_LEG_LANDED', 'R_DISTANCE_ATT',
       'R_DISTANCE_LANDED', 'B_DISTANCE_ATT', 'B_DISTANCE_LANDED',
       'R_CLINCH_ATT', 'R_CLINCH_LANDED', 'B_CLINCH_ATT', 'B_CLINCH_LANDED',
       'R_GROUND_ATT', 'R_GROUND_LANDED', 'B_GROUND_ATT', 'B_GROUND_LANDED',
       'title_bout', 'weight_class', 'total_time_fought(sec)', 'no_of_rounds',
       'Red_win']]


# In[302]:


df_fighter_history_train = df_fighter_history[['fighter_name', 'Height',
                            'Weight', 'Reach', 'Stance', 'DOB', 'win_by',
                            'date',
                           'win']]


# In[303]:


df_fighter_history_train.head(3)


# - first merge for red fighter

# In[304]:


df_train = pd.merge(df_fights_train, df_fighter_history_train, left_on='R_fighter',right_on='fighter_name',sort=False)


# - for each bout, filter to only previous fights

# In[305]:


df_train = df_train[df_train['date_x'] > df_train['date_y']]
df_train.drop(columns=['date_y','fighter_name'], inplace=True)


# - create dummy variables for fighter-specific categorical variables (i.e. stance, win_by)

# In[306]:


df_train = pd.concat([df_train, pd.get_dummies(df_train['win_by_y'])], axis=1)
df_train.drop(columns=['win_by_y','Other'], inplace=True)
df_train.rename(columns={'date_x':'date', 'KO':'R_KO_win_%', 'Submission':'R_Sub_win_%'}, inplace=True)

df_train = pd.concat([df_train, pd.get_dummies(df_train['Stance'])], axis=1)
df_train.drop(columns=['Stance','Switch','Open Stance','Sideways'], inplace=True)
df_train.rename(columns={'Orthodox':'R_Stance_Orthodox',
                         'Southpaw':'R_Stance_Southpaw',
                         'Height':'R_Height',
                         'Weight':'R_Weight',
                         'Reach':'R_Reach'}, inplace=True)


# - recalculate number of past fights, fighter record, and fighter age

# In[307]:


df_train['R_num_fights'] = df_train.groupby(['R_fighter','date'])['date'].transform('count')

df_train['R_num_wins'] = df_train.groupby(['R_fighter','date'])['win'].transform('sum')

df_train['R_record'] = df_train['R_num_wins']/df_train['R_num_fights']

df_train.drop(columns=['win','R_num_wins'], inplace=True)


# In[308]:


df_train['R_age'] = df_train['date'] - df_train['DOB']
df_train['R_age']=df_train['R_age']/np.timedelta64(1,'Y')
df_train['R_age'].fillna(df_train['R_age'].mean(), inplace=True)

df_train.drop(columns=['DOB'], inplace=True)


# In[309]:


df_train[['R_KO_win_%', 'R_Sub_win_%']] = df_train.groupby(['R_fighter', 'B_fighter', 'R_KD', 'B_KD', 'R_SIG_STR_pct',
       'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct', 'R_SUB_ATT', 'B_SUB_ATT',
       'R_PASS', 'B_PASS', 'R_REV', 'B_REV', 'win_by_x', 'last_round', 'date',
       'location', 'R_SIG_STR._ATT', 'R_SIG_STR._LANDED', 'B_SIG_STR._ATT',
       'B_SIG_STR._LANDED', 'R_TOTAL_STR._ATT', 'R_TOTAL_STR._LANDED',
       'B_TOTAL_STR._ATT', 'B_TOTAL_STR._LANDED', 'R_TD_ATT', 'R_TD_LANDED',
       'B_TD_ATT', 'B_TD_LANDED', 'R_HEAD_ATT', 'R_HEAD_LANDED', 'B_HEAD_ATT',
       'B_HEAD_LANDED', 'R_BODY_ATT', 'R_BODY_LANDED', 'B_BODY_ATT',
       'B_BODY_LANDED', 'R_LEG_ATT', 'R_LEG_LANDED', 'B_LEG_ATT',
       'B_LEG_LANDED', 'R_DISTANCE_ATT', 'R_DISTANCE_LANDED', 'B_DISTANCE_ATT',
       'B_DISTANCE_LANDED', 'R_CLINCH_ATT', 'R_CLINCH_LANDED', 'B_CLINCH_ATT',
       'B_CLINCH_LANDED', 'R_GROUND_ATT', 'R_GROUND_LANDED', 'B_GROUND_ATT',
       'B_GROUND_LANDED', 'title_bout', 'weight_class',
       'total_time_fought(sec)', 'no_of_rounds', 'Red_win', 'R_Height', 'R_Weight',
       'R_Reach', 'R_Stance_Orthodox',
       'R_Stance_Southpaw', 'R_num_fights', 'R_record', 'R_age'])['R_KO_win_%', 'R_Sub_win_%'].transform('mean')

df_train = df_train.drop_duplicates()


# - repeat steps for blue fighter

# In[ ]:


#merge blue fighters
df_train = pd.merge(df_train, df_fighter_history_train, left_on='B_fighter',right_on='fighter_name',sort=False)

#only past fights
df_train = df_train[df_train['date_x'] > df_train['date_y']]
df_train.drop(columns=['date_y','fighter_name'], inplace=True)

#dummy variables
df_train = pd.concat([df_train, pd.get_dummies(df_train['win_by'])], axis=1)
df_train.drop(columns=['win_by','Other'], inplace=True)
df_train.rename(columns={'date_x':'date', 'KO':'B_KO_win_%', 'Submission':'B_Sub_win_%'}, inplace=True)

df_train = pd.concat([df_train, pd.get_dummies(df_train['Stance'])], axis=1)
df_train.drop(columns=['Stance','Switch','Open Stance','Sideways'], inplace=True)
df_train.rename(columns={'Orthodox':'B_Stance_Orthodox',
                         'Southpaw':'B_Stance_Southpaw',
                         'Height':'B_Height',
                         'Weight':'B_Weight',
                         'Reach':'B_Reach'}, inplace=True)

#num_fights and record
df_train['B_num_fights'] = df_train.groupby(['B_fighter','date'])['date'].transform('count')

df_train['B_num_wins'] = df_train.groupby(['B_fighter','date'])['win'].transform('sum')

df_train['B_record'] = df_train['B_num_wins']/df_train['B_num_fights']

df_train.drop(columns=['win','B_num_wins'], inplace=True)

#age
df_train['B_age'] = df_train['date'] - df_train['DOB']
df_train['B_age']=df_train['B_age']/np.timedelta64(1,'Y')
df_train['B_age'].fillna(df_train['B_age'].mean(), inplace=True)

df_train.drop(columns=['DOB'], inplace=True)

#win_by percentages
df_train[['B_KO_win_%', 'B_Sub_win_%']] = df_train.groupby(['R_fighter', 'B_fighter', 'R_KD', 'B_KD', 'R_SIG_STR_pct',
       'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct', 'R_SUB_ATT', 'B_SUB_ATT',
       'R_PASS', 'B_PASS', 'R_REV', 'B_REV', 'win_by_x', 'last_round', 'date',
       'location', 'R_SIG_STR._ATT', 'R_SIG_STR._LANDED', 'B_SIG_STR._ATT',
       'B_SIG_STR._LANDED', 'R_TOTAL_STR._ATT', 'R_TOTAL_STR._LANDED',
       'B_TOTAL_STR._ATT', 'B_TOTAL_STR._LANDED', 'R_TD_ATT', 'R_TD_LANDED',
       'B_TD_ATT', 'B_TD_LANDED', 'R_HEAD_ATT', 'R_HEAD_LANDED', 'B_HEAD_ATT',
       'B_HEAD_LANDED', 'R_BODY_ATT', 'R_BODY_LANDED', 'B_BODY_ATT',
       'B_BODY_LANDED', 'R_LEG_ATT', 'R_LEG_LANDED', 'B_LEG_ATT',
       'B_LEG_LANDED', 'R_DISTANCE_ATT', 'R_DISTANCE_LANDED', 'B_DISTANCE_ATT',
       'B_DISTANCE_LANDED', 'R_CLINCH_ATT', 'R_CLINCH_LANDED', 'B_CLINCH_ATT',
       'B_CLINCH_LANDED', 'R_GROUND_ATT', 'R_GROUND_LANDED', 'B_GROUND_ATT',
       'B_GROUND_LANDED', 'title_bout', 'weight_class',
       'total_time_fought(sec)', 'no_of_rounds', 'Red_win', 'R_Height',
       'R_Weight', 'R_Reach', 'R_KO_win_%', 'R_Sub_win_%', 'R_Stance_Orthodox',
       'R_Stance_Southpaw', 'R_num_fights', 'R_record', 'R_age', 'B_Height',
       'B_Weight', 'B_Reach', 'B_Stance_Orthodox',
       'B_Stance_Southpaw', 'B_num_fights', 'B_record', 'B_age'])['B_KO_win_%', 'B_Sub_win_%'].transform('mean')

df_train = df_train.drop_duplicates()


# In[311]:


df_train[df_train['R_fighter']=='Jon Jones']


# - create dummy variables for weight class

# In[312]:


df_train.drop(columns=['win_by_x','last_round'], inplace=True)


# In[313]:


df_train = pd.concat([df_train, pd.get_dummies(df_train['weight_class'])], axis=1)
df_train.drop(columns=['weight_class','Open Weight','Catch Weight'], inplace=True)


# In[314]:


df_train[df_train['R_fighter']=='Kamaru Usman']


# - export relevant data frames for further use

# In[315]:


df_fights.to_csv(DATA_PATH+'/df_fights.csv', index = False, header=True)
df_fighter_history.to_csv(DATA_PATH+'/df_fighter_history.csv', index = False, header=True)
df_fights_train.to_csv(DATA_PATH+'/df_fights_train.csv', index = False, header=True)
df_fighter_history_train.to_csv(DATA_PATH+'/df_fighter_history_train.csv', index = False, header=True)
df_train.to_csv(DATA_PATH+'/df_train.csv', index = False, header=True)
