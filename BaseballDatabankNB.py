
# coding: utf-8

# In[ ]:


# load BaseballDataBank.py
import pandas as pd
import glob, os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# ## Reading in BaseballDatabank csv files


# In[ ]:


def read_all_databank_core_csv(directory):
    """
    read all csv files in the specified baseball databank directory and
    populate a dictionary storing each of the tables keyed to its name
    """
    dfs = {}
    files = glob.glob('{}/*.csv'.format(directory))
    for f in files:
        d, name = os.path.split(f)
        table = os.path.splitext(name)[0]
        df = pd.read_csv(f)
        dfs[table] = df
    return dfs

bbdfs = read_all_databank_core_csv('baseballdatabank/core/')

# extract a few for further processing
batting = bbdfs['Batting']
pitching = bbdfs['Pitching']
teams = bbdfs['Teams']


# ## Taking a peek
# ### batting is year-by-year for each individual player
# ### teams is year-by-year for each team

# In[ ]:


pd.options.display.max_colwidth = 350


# In[ ]:


batting.head()


# In[ ]:


pitching.columns


# In[ ]:


teams.head()


# ## Filtering data (Hank Aaron's year-by-year batting statistics)

# In[ ]:


batting[batting.playerID=='aaronha01']


# ## Aggregating data (Hank Aaron's career batting statistics)

# In[ ]:


batting[batting.playerID=='aaronha01'].sum(numeric_only=True).drop(['yearID']).astype(int)


# ## Adding derived data: calculating singles (1B) from H,2B,3B,HR

# In[ ]:


batting['1B'] = batting['H'] - batting['2B'] - batting['3B'] - batting['HR']
teams['1B'] = teams['H'] - teams['2B'] - teams['3B'] - teams['HR']
batting.head()


# ## A succinct history of hitting in baseball
# ### (time progresses from light to dark)

# In[ ]:


batting_by_year = batting.groupby('yearID').sum().reset_index()
hit_vars = ['1B', '2B', '3B', 'HR', 'SO', 'BB']


# In[ ]:


#pg = sns.pairplot(batting_by_year, size=2, vars=hit_vars, hue='yearID', palette='Blues')
#pg.hue_names = ["_nolegend_"]
g = sns.PairGrid(batting_by_year, vars=hit_vars, hue='yearID', palette='Blues')
g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
# g = g.map_diag(plt.hist, edgecolor="w")


# In[ ]:


pg = sns.pairplot(batting_by_year, height=2, vars=hit_vars, hue='yearID', palette='Blues')


# ## The correlation of hitting statistics

# In[ ]:


sns.heatmap(batting_by_year[hit_vars].corr(), annot=True)


# ## Grouping batting data by player

# In[ ]:


pl_bat = batting.groupby('playerID').sum().reset_index()
bbdfs['CareerBatting'] = pl_bat
pl_bat.head()


# ## Adding more derived data: The Slash Line (BA / OBP / SLG)

# In[ ]:


pl_bat['BA']= pl_bat['H'] / pl_bat['AB']
pl_bat['OBP'] = (pl_bat['H']+pl_bat['BB']+pl_bat['HBP']) / (pl_bat['AB']+pl_bat['BB']+pl_bat['HBP']+pl_bat['SF'])
pl_bat['SLG'] = (pl_bat['1B']+2*pl_bat['2B']+3*pl_bat['3B']+4*pl_bat['HR']) / pl_bat['AB']
pl_bat.head()


# ## Filtering data part 2
# ### Top all-time slugging percentages (at least 100 AB)

# In[ ]:


pl_bat[pl_bat.AB >= 100].sort_values(by='SLG', ascending=False).head(30)


# ## Sabermetrics: the "Pythagorean" theorem of baseball

# In[ ]:


teams['Observed_WinRatio'] = teams.W/teams.G
teams['Expected_WinRatio_183'] = 1 / (1 + (teams.RA/teams.R)**1.83)
teams['Expected_WinRatio_2'] = 1 / (1 + (teams.RA/teams.R)**2)
teams['Overachieving'] = teams['Observed_WinRatio']/teams['Expected_WinRatio_183']
teams['Fraction_Runs'] = teams.R/(teams.R + teams.RA)
teams.plot.scatter('Expected_WinRatio_2', 'Observed_WinRatio')



# ## Writing all the dataframes to a SQL database

# In[ ]:


def write_all_tables_to_sqlite(dfs, sql_filename):
    engine = create_engine('sqlite:///{}'.format(sql_filename))
    for table, df in dfs.items():
        df.to_sql(table, con=engine, index=False)
    engine.dispose()
    
sqlite_filename = 'bbdb.sqlite'
try:
    os.remove(sqlite_filename)
except FileNotFoundError:
    pass
write_all_tables_to_sqlite(bbdfs, sqlite_filename)



# ## Make SQL query to Baseball DB

# In[ ]:


engine = create_engine('sqlite:///bbdb.sqlite')

top_slugging = pd.read_sql_query('select * from CareerBatting where AB>= 100 order by SLG desc limit 30', engine)
top_slugging


# ## More history: the saga of Home Runs

# In[ ]:


ax = batting_by_year.plot('yearID', 'HR', figsize=(16,8))
annot1920 = plt.text(1920, 0, '<End of dead ball era')
annot1942 = plt.text(1942, 400, '<WW2')
annot1961 = plt.text(1961, 1200, '<3 teams added to AL')
annot1962 = plt.text(1962, 1400, '<2 teams added to NL')
annot1969 = plt.text(1969, 1600, '<2 teams added to both AL + NL')
annot1995 = plt.text(1995, 3000, '<Steroids rampant')
annot2003 = plt.text(2003, 3500, '<Steroids tested for')
annot2015 = plt.text(2015, 4000, '<Fascination with launch angle / strikeouts be damned')


# In[ ]:


ax = batting_by_year.plot('yearID', 'HR', figsize=(16,8))


# ## Correcting for demographics: HR per AB

# In[ ]:


batting_by_year = batting.groupby('yearID').sum().reset_index()
batting_by_year.set_index('yearID', inplace=True)
batting_by_year_perAB = batting_by_year.div(batting_by_year.AB, axis=0).reset_index()
batting_by_year.reset_index(inplace=True)


# In[ ]:


ax = batting_by_year_perAB.plot('yearID', 'HR', figsize=(16,8))
annot1920 = plt.text(1920, 0.001, '<End of dead ball era')
annot1942 = plt.text(1942, 0.009, '<WW2')
annot1961 = plt.text(1961, 0.013, '<3 teams added to AL')
annot1962 = plt.text(1962, 0.014, '<2 teams added to NL')
annot1969 = plt.text(1969, 0.015, '<2 teams added to both AL + NL')
annot1995 = plt.text(1995, 0.020, '<Steroids rampant')
annot2003 = plt.text(2003, 0.0225, '<Steroids tested for')
annot2015 = plt.text(2015, 0.025, '<Fascination with launch angle / strikeouts be damned')


# In[ ]:


pitching_by_year = pitching.groupby('yearID').sum().reset_index()


# In[ ]:


pitching_by_year.set_index('yearID', inplace=True)


# In[ ]:


pitching_by_year_perIPouts = pitching_by_year.div(pitching_by_year.IPouts, axis=0).reset_index()


# In[ ]:


pitching_by_year_perIPouts


# In[ ]:


pitching_by_year.reset_index(inplace=True)


# In[ ]:


ax = pitching_by_year_perIPouts.plot('yearID', 'SO', figsize=(16,8))
annot1920 = plt.text(1920, 0.05, '<End of dead ball era')
annot1942 = plt.text(1942, 0.09, '<WW2')
annot1961 = plt.text(1961, 0.13, '<3 teams added to AL')
annot1962 = plt.text(1962, 0.14, '<2 teams added to NL')
annot1969 = plt.text(1969, 0.15, '<2 teams added to both AL + NL')
annot1995 = plt.text(1995, 0.20, '<Steroids rampant')
annot2003 = plt.text(2003, 0.225, '<Steroids tested for')
annot2015 = plt.text(2015, 0.25, '<Fascination with launch angle / strikeouts be damned')

