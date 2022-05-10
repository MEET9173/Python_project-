import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px
from tkinter import *

win= Tk()
win.geometry("750x250")

df = pd.read_csv('netflix_titles.csv')

df.head(3)

# Missing data

for i in df.columns:
    null_rate = df[i].isna().sum() / len(df) * 100 
    if null_rate > 0 :
        print("{} null rate: {}%".format(i,round(null_rate,2)))
        
df['country'] = df['country'].fillna(df['country'].mode()[0])


df['cast'].replace(np.nan, 'No Data',inplace  = True)
df['director'].replace(np.nan, 'No Data',inplace  = True)

# Drops

df.dropna(inplace=True)

# Drop Duplicates

df.drop_duplicates(inplace= True)

df.isnull().sum()

df.info()

df["date_added"] = pd.to_datetime(df['date_added'])

df['month_added']=df['date_added'].dt.month
df['month_name_added']=df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year




from sklearn.preprocessing import MultiLabelBinarizer 

import matplotlib.colors


# Custom colour map based on Netflix palette
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710','#f5f5f1'])


def graph():
    
    def genre_heatmap(df, title):
        df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
        Types = []
        for i in df['genre']: Types += i
        Types = set(Types)
        print("There are {} types in the Netflix {} Dataset".format(len(Types),title))    
        test = df['genre']
        mlb = MultiLabelBinarizer()
        res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
        corr = res.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.text(.54,.88,'Genre correlation', fontfamily='serif',fontweight='bold',fontsize=15)
        fig.text(.75,.665,
                '''
                 It is interesting that Independant Movies
                 tend to be Dramas. 
                 
                 Another observation is that 
                 Internatinal Movies are rarely
                 in the Children's genre.
                 ''', fontfamily='serif',fontsize=12,ha='right')
        pl = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, vmin=-.3, center=0, square=True, linewidths=2.5)
        
        plt.show()
        
    # =============================================================================

        
    df_tv = df[df["type"] == "TV Show"]
    df_movies = df[df["type"] == "Movie"]
    
    
    genre_heatmap(df_movies, 'Movie')
    plt.show()
    plt.write_image("images/fig1.png")

Button(win, text= "Show Graph", command= graph).pack(pady=20)
win.mainloop()