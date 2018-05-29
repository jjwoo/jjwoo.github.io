

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import wordnet

import warnings; warnings.simplefilter('ignore')
```


```python
from surprise import Reader, Dataset, SVD, evaluate
```

# EDA


```python
# Load the data.
movies = pd.read_csv("./movies-dataset/movies_metadata.csv")
movies.head(20)
```

    /Users/jjwoo/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2698: DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>1995-12-15</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[{'id': 10749, 'name': 'Romance'}, {'id': 35, ...</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>81452156.0</td>
      <td>127.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>1995-02-10</td>
      <td>76578911.0</td>
      <td>106.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>NaN</td>
      <td>60000000</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 80, 'nam...</td>
      <td>NaN</td>
      <td>949</td>
      <td>tt0113277</td>
      <td>en</td>
      <td>Heat</td>
      <td>Obsessive master thief, Neil McCauley leads a ...</td>
      <td>...</td>
      <td>1995-12-15</td>
      <td>187436818.0</td>
      <td>170.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>A Los Angeles Crime Saga</td>
      <td>Heat</td>
      <td>False</td>
      <td>7.7</td>
      <td>1886.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>NaN</td>
      <td>58000000</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 10749, '...</td>
      <td>NaN</td>
      <td>11860</td>
      <td>tt0114319</td>
      <td>en</td>
      <td>Sabrina</td>
      <td>An ugly duckling having undergone a remarkable...</td>
      <td>...</td>
      <td>1995-12-15</td>
      <td>0.0</td>
      <td>127.0</td>
      <td>[{'iso_639_1': 'fr', 'name': 'Français'}, {'is...</td>
      <td>Released</td>
      <td>You are cordially invited to the most surprisi...</td>
      <td>Sabrina</td>
      <td>False</td>
      <td>6.2</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 12, 'nam...</td>
      <td>NaN</td>
      <td>45325</td>
      <td>tt0112302</td>
      <td>en</td>
      <td>Tom and Huck</td>
      <td>A mischievous young boy, Tom Sawyer, witnesses...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>0.0</td>
      <td>97.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>The Original Bad Boys.</td>
      <td>Tom and Huck</td>
      <td>False</td>
      <td>5.4</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>NaN</td>
      <td>35000000</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 12, 'nam...</td>
      <td>NaN</td>
      <td>9091</td>
      <td>tt0114576</td>
      <td>en</td>
      <td>Sudden Death</td>
      <td>International action superstar Jean Claude Van...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>64350171.0</td>
      <td>106.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Terror goes into overtime.</td>
      <td>Sudden Death</td>
      <td>False</td>
      <td>5.5</td>
      <td>174.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>{'id': 645, 'name': 'James Bond Collection', '...</td>
      <td>58000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 28, '...</td>
      <td>http://www.mgm.com/view/movie/757/Goldeneye/</td>
      <td>710</td>
      <td>tt0113189</td>
      <td>en</td>
      <td>GoldenEye</td>
      <td>James Bond must unmask the mysterious head of ...</td>
      <td>...</td>
      <td>1995-11-16</td>
      <td>352194034.0</td>
      <td>130.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>No limits. No fears. No substitutes.</td>
      <td>GoldenEye</td>
      <td>False</td>
      <td>6.6</td>
      <td>1194.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>False</td>
      <td>NaN</td>
      <td>62000000</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>
      <td>NaN</td>
      <td>9087</td>
      <td>tt0112346</td>
      <td>en</td>
      <td>The American President</td>
      <td>Widowed U.S. president Andrew Shepherd, one of...</td>
      <td>...</td>
      <td>1995-11-17</td>
      <td>107879496.0</td>
      <td>106.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Why can't the most powerful man in the world h...</td>
      <td>The American President</td>
      <td>False</td>
      <td>6.5</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 27, 'nam...</td>
      <td>NaN</td>
      <td>12110</td>
      <td>tt0112896</td>
      <td>en</td>
      <td>Dracula: Dead and Loving It</td>
      <td>When a lawyer shows up at the vampire's doorst...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>0.0</td>
      <td>88.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Dracula: Dead and Loving It</td>
      <td>False</td>
      <td>5.7</td>
      <td>210.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>False</td>
      <td>{'id': 117693, 'name': 'Balto Collection', 'po...</td>
      <td>0</td>
      <td>[{'id': 10751, 'name': 'Family'}, {'id': 16, '...</td>
      <td>NaN</td>
      <td>21032</td>
      <td>tt0112453</td>
      <td>en</td>
      <td>Balto</td>
      <td>An outcast half-wolf risks his life to prevent...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>11348324.0</td>
      <td>78.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Part Dog. Part Wolf. All Hero.</td>
      <td>Balto</td>
      <td>False</td>
      <td>7.1</td>
      <td>423.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>False</td>
      <td>NaN</td>
      <td>44000000</td>
      <td>[{'id': 36, 'name': 'History'}, {'id': 18, 'na...</td>
      <td>NaN</td>
      <td>10858</td>
      <td>tt0113987</td>
      <td>en</td>
      <td>Nixon</td>
      <td>An all-star cast powers this epic look at Amer...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>13681765.0</td>
      <td>192.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Triumphant in Victory, Bitter in Defeat. He Ch...</td>
      <td>Nixon</td>
      <td>False</td>
      <td>7.1</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>False</td>
      <td>NaN</td>
      <td>98000000</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 12, 'nam...</td>
      <td>NaN</td>
      <td>1408</td>
      <td>tt0112760</td>
      <td>en</td>
      <td>Cutthroat Island</td>
      <td>Morgan Adams and her slave, William Shaw, are ...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>10017322.0</td>
      <td>119.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>The Course Has Been Set. There Is No Turning B...</td>
      <td>Cutthroat Island</td>
      <td>False</td>
      <td>5.7</td>
      <td>137.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>False</td>
      <td>NaN</td>
      <td>52000000</td>
      <td>[{'id': 18, 'name': 'Drama'}, {'id': 80, 'name...</td>
      <td>NaN</td>
      <td>524</td>
      <td>tt0112641</td>
      <td>en</td>
      <td>Casino</td>
      <td>The life of the gambling paradise – Las Vegas ...</td>
      <td>...</td>
      <td>1995-11-22</td>
      <td>116112375.0</td>
      <td>178.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>No one stays at the top forever.</td>
      <td>Casino</td>
      <td>False</td>
      <td>7.8</td>
      <td>1343.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>False</td>
      <td>NaN</td>
      <td>16500000</td>
      <td>[{'id': 18, 'name': 'Drama'}, {'id': 10749, 'n...</td>
      <td>NaN</td>
      <td>4584</td>
      <td>tt0114388</td>
      <td>en</td>
      <td>Sense and Sensibility</td>
      <td>Rich Mr. Dashwood dies, leaving his second wif...</td>
      <td>...</td>
      <td>1995-12-13</td>
      <td>135000000.0</td>
      <td>136.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Lose your heart and come to your senses.</td>
      <td>Sense and Sensibility</td>
      <td>False</td>
      <td>7.2</td>
      <td>364.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>False</td>
      <td>NaN</td>
      <td>4000000</td>
      <td>[{'id': 80, 'name': 'Crime'}, {'id': 35, 'name...</td>
      <td>NaN</td>
      <td>5</td>
      <td>tt0113101</td>
      <td>en</td>
      <td>Four Rooms</td>
      <td>It's Ted the Bellhop's first night on the job....</td>
      <td>...</td>
      <td>1995-12-09</td>
      <td>4300000.0</td>
      <td>98.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Twelve outrageous guests. Four scandalous requ...</td>
      <td>Four Rooms</td>
      <td>False</td>
      <td>6.5</td>
      <td>539.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>False</td>
      <td>{'id': 3167, 'name': 'Ace Ventura Collection',...</td>
      <td>30000000</td>
      <td>[{'id': 80, 'name': 'Crime'}, {'id': 35, 'name...</td>
      <td>NaN</td>
      <td>9273</td>
      <td>tt0112281</td>
      <td>en</td>
      <td>Ace Ventura: When Nature Calls</td>
      <td>Summoned from an ashram in Tibet, Ace finds hi...</td>
      <td>...</td>
      <td>1995-11-10</td>
      <td>212385533.0</td>
      <td>90.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>New animals. New adventures. Same hair.</td>
      <td>Ace Ventura: When Nature Calls</td>
      <td>False</td>
      <td>6.1</td>
      <td>1128.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>False</td>
      <td>NaN</td>
      <td>60000000</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 35, 'nam...</td>
      <td>NaN</td>
      <td>11517</td>
      <td>tt0113845</td>
      <td>en</td>
      <td>Money Train</td>
      <td>A vengeful New York transit cop decides to ste...</td>
      <td>...</td>
      <td>1995-11-21</td>
      <td>35431113.0</td>
      <td>103.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Get on, or GET OUT THE WAY!</td>
      <td>Money Train</td>
      <td>False</td>
      <td>5.4</td>
      <td>224.0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 24 columns</p>
</div>



For release_date, simplify to only Year for easier manipulability


```python
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
```

### Drop obviously irrelevant columns


```python
movies.columns
```




    Index(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
           'imdb_id', 'original_language', 'original_title', 'overview',
           'popularity', 'poster_path', 'production_companies',
           'production_countries', 'release_date', 'revenue', 'runtime',
           'spoken_languages', 'status', 'tagline', 'title', 'video',
           'vote_average', 'vote_count', 'year'],
          dtype='object')




```python
movies = movies.drop(['adult','belongs_to_collection','homepage','poster_path','production_countries','tagline','video'], axis=1)
```


```python
movies.columns
```




    Index(['budget', 'genres', 'id', 'imdb_id', 'original_language',
           'original_title', 'overview', 'popularity', 'production_companies',
           'release_date', 'revenue', 'runtime', 'spoken_languages', 'status',
           'title', 'vote_average', 'vote_count', 'year'],
          dtype='object')




```python
movies.isnull().sum()
```




    budget                    0
    genres                    0
    id                        0
    imdb_id                  17
    original_language        11
    original_title            0
    overview                954
    popularity                5
    production_companies      3
    release_date             87
    revenue                   6
    runtime                 263
    spoken_languages          6
    status                   87
    title                     6
    vote_average              6
    vote_count                6
    year                      0
    dtype: int64




```python
movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45466 entries, 0 to 45465
    Data columns (total 18 columns):
    budget                  45466 non-null object
    genres                  45466 non-null object
    id                      45466 non-null object
    imdb_id                 45449 non-null object
    original_language       45455 non-null object
    original_title          45466 non-null object
    overview                44512 non-null object
    popularity              45461 non-null object
    production_companies    45463 non-null object
    release_date            45379 non-null object
    revenue                 45460 non-null float64
    runtime                 45203 non-null float64
    spoken_languages        45460 non-null object
    status                  45379 non-null object
    title                   45460 non-null object
    vote_average            45460 non-null float64
    vote_count              45460 non-null float64
    year                    45466 non-null object
    dtypes: float64(4), object(14)
    memory usage: 6.2+ MB



```python
movies.shape
```




    (45466, 18)



# Baseline Model: Simple Recommender

The Simple Recommender is a baseline model which offers simple recommendations based on movie's weighted rating. Then, the top few selected movies will be displayed. 

In addition, I will pass in a genre argument for extra option for the user.


```python
# extract the genres
movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
```


```python
movies.genres.head(5)
```




    0     [Animation, Comedy, Family]
    1    [Adventure, Fantasy, Family]
    2               [Romance, Comedy]
    3        [Comedy, Drama, Romance]
    4                        [Comedy]
    Name: genres, dtype: object



I will use a weighted rating that takes into account the average rating and the number of votes for the movie. Hence, a movie that has a 8 rating from 50,000 voters will have a higher score than a movie with the same rating but with fewer voters.

Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$

where,
* *v* is the number of votes for the movie
* *m* is the minimum votes required to be listed in the chart
* *R* is the average rating of the movie
* *C* is the mean vote across the whole report


```python
# Calculate C
C = movies['vote_average'].mean()
print (C)
```

    5.618207215133889


The average rating of a movie is around 5.6, on a scale of 10.

For m, it is an arbitrary number where we will filter out movies that have few votes.

We will use 90th percentile as a cutoff. Hence, for a movie to feature in the list, it has to have more votes than 90% of the movies in the list.


```python
# Minimum number of votes required to be in the chart
m = movies['vote_count'].quantile(0.90)
print(m)
```

    160.0



```python
# Filter out all qualified movies into a new df
q_movies = movies.copy().loc[movies['vote_count'] >= m]
q_movies.shape
```




    (4555, 18)



There are 4555 movies which qualify to be in this list.


```python
q_movies.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.9469</td>
      <td>[{'name': 'Pixar Animation Studios', 'id': 3}]</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Toy Story</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>17.0155</td>
      <td>[{'name': 'TriStar Pictures', 'id': 559}, {'na...</td>
      <td>1995-12-15</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Jumanji</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[Comedy]</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>8.38752</td>
      <td>[{'name': 'Sandollar Productions', 'id': 5842}...</td>
      <td>1995-02-10</td>
      <td>76578911.0</td>
      <td>106.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Father of the Bride Part II</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compute the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
```


```python
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
```

### Top movies by weighted rating


```python
#Sort movies based on score calculated
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score','genres']].head(15)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>score</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>314</th>
      <td>The Shawshank Redemption</td>
      <td>8358.0</td>
      <td>8.5</td>
      <td>8.445869</td>
      <td>[Drama, Crime]</td>
    </tr>
    <tr>
      <th>834</th>
      <td>The Godfather</td>
      <td>6024.0</td>
      <td>8.5</td>
      <td>8.425439</td>
      <td>[Drama, Crime]</td>
    </tr>
    <tr>
      <th>10309</th>
      <td>Dilwale Dulhania Le Jayenge</td>
      <td>661.0</td>
      <td>9.1</td>
      <td>8.421453</td>
      <td>[Comedy, Drama, Romance]</td>
    </tr>
    <tr>
      <th>12481</th>
      <td>The Dark Knight</td>
      <td>12269.0</td>
      <td>8.3</td>
      <td>8.265477</td>
      <td>[Drama, Action, Crime, Thriller]</td>
    </tr>
    <tr>
      <th>2843</th>
      <td>Fight Club</td>
      <td>9678.0</td>
      <td>8.3</td>
      <td>8.256385</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>292</th>
      <td>Pulp Fiction</td>
      <td>8670.0</td>
      <td>8.3</td>
      <td>8.251406</td>
      <td>[Thriller, Crime]</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Schindler's List</td>
      <td>4436.0</td>
      <td>8.3</td>
      <td>8.206639</td>
      <td>[Drama, History, War]</td>
    </tr>
    <tr>
      <th>23673</th>
      <td>Whiplash</td>
      <td>4376.0</td>
      <td>8.3</td>
      <td>8.205404</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>5481</th>
      <td>Spirited Away</td>
      <td>3968.0</td>
      <td>8.3</td>
      <td>8.196055</td>
      <td>[Fantasy, Adventure, Animation, Family]</td>
    </tr>
    <tr>
      <th>2211</th>
      <td>Life Is Beautiful</td>
      <td>3643.0</td>
      <td>8.3</td>
      <td>8.187171</td>
      <td>[Comedy, Drama]</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>The Godfather: Part II</td>
      <td>3418.0</td>
      <td>8.3</td>
      <td>8.180076</td>
      <td>[Drama, Crime]</td>
    </tr>
    <tr>
      <th>1152</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>3001.0</td>
      <td>8.3</td>
      <td>8.164256</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Forrest Gump</td>
      <td>8147.0</td>
      <td>8.2</td>
      <td>8.150272</td>
      <td>[Comedy, Drama, Romance]</td>
    </tr>
    <tr>
      <th>1154</th>
      <td>The Empire Strikes Back</td>
      <td>5998.0</td>
      <td>8.2</td>
      <td>8.132919</td>
      <td>[Adventure, Action, Science Fiction]</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>Psycho</td>
      <td>2405.0</td>
      <td>8.3</td>
      <td>8.132715</td>
      <td>[Drama, Horror, Thriller]</td>
    </tr>
  </tbody>
</table>
</div>



### Top movies by genre


```python
# To split the movies into one genre per row
s = movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
genre_movies = movies.drop('genres', axis=1).join(s)

s.head(8)
```




    0    Animation
    0       Comedy
    0       Family
    1    Adventure
    1      Fantasy
    1       Family
    2      Romance
    2       Comedy
    Name: genre, dtype: object




```python
genre_movies.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30000000</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.9469</td>
      <td>[{'name': 'Pixar Animation Studios', 'id': 3}]</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Toy Story</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>0</th>
      <td>30000000</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.9469</td>
      <td>[{'name': 'Pixar Animation Studios', 'id': 3}]</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Toy Story</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
def genre_chart(genre, percentile=0.90):
    df = genre_movies[genre_movies['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
#     C = vote_averages.mean()
#     m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    # qualified['score'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified['score'] = q_movies['score']
    qualified = qualified.sort_values('score', ascending=False).head(250)
    
    return qualified
```


```python
genre_chart('War').head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>year</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>popularity</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>522</th>
      <td>Schindler's List</td>
      <td>1993</td>
      <td>4436</td>
      <td>8</td>
      <td>41.7251</td>
      <td>8.206639</td>
    </tr>
    <tr>
      <th>24860</th>
      <td>The Imitation Game</td>
      <td>2014</td>
      <td>5895</td>
      <td>8</td>
      <td>31.5959</td>
      <td>7.937062</td>
    </tr>
    <tr>
      <th>5857</th>
      <td>The Pianist</td>
      <td>2002</td>
      <td>1927</td>
      <td>8</td>
      <td>14.8116</td>
      <td>7.909733</td>
    </tr>
    <tr>
      <th>13605</th>
      <td>Inglourious Basterds</td>
      <td>2009</td>
      <td>6598</td>
      <td>7</td>
      <td>16.8956</td>
      <td>7.845977</td>
    </tr>
    <tr>
      <th>5553</th>
      <td>Grave of the Fireflies</td>
      <td>1988</td>
      <td>974</td>
      <td>8</td>
      <td>0.010902</td>
      <td>7.835726</td>
    </tr>
    <tr>
      <th>1165</th>
      <td>Apocalypse Now</td>
      <td>1979</td>
      <td>2112</td>
      <td>8</td>
      <td>13.5963</td>
      <td>7.832268</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>Saving Private Ryan</td>
      <td>1998</td>
      <td>5148</td>
      <td>7</td>
      <td>21.7581</td>
      <td>7.831220</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>Full Metal Jacket</td>
      <td>1987</td>
      <td>2595</td>
      <td>7</td>
      <td>13.9415</td>
      <td>7.767482</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>1964</td>
      <td>1472</td>
      <td>8</td>
      <td>9.80398</td>
      <td>7.766491</td>
    </tr>
    <tr>
      <th>43190</th>
      <td>Band of Brothers</td>
      <td>2001</td>
      <td>725</td>
      <td>8</td>
      <td>7.903731</td>
      <td>7.733235</td>
    </tr>
  </tbody>
</table>
</div>




```python
genre_chart('Romance').head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>year</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>popularity</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10309</th>
      <td>Dilwale Dulhania Le Jayenge</td>
      <td>1995</td>
      <td>661</td>
      <td>9</td>
      <td>34.457</td>
      <td>8.421453</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Forrest Gump</td>
      <td>1994</td>
      <td>8147</td>
      <td>8</td>
      <td>48.3072</td>
      <td>8.150272</td>
    </tr>
    <tr>
      <th>40251</th>
      <td>Your Name.</td>
      <td>2016</td>
      <td>1030</td>
      <td>8</td>
      <td>34.461252</td>
      <td>8.112532</td>
    </tr>
    <tr>
      <th>40882</th>
      <td>La La Land</td>
      <td>2016</td>
      <td>4745</td>
      <td>7</td>
      <td>19.681686</td>
      <td>7.825568</td>
    </tr>
    <tr>
      <th>22168</th>
      <td>Her</td>
      <td>2013</td>
      <td>4215</td>
      <td>7</td>
      <td>13.8295</td>
      <td>7.816552</td>
    </tr>
    <tr>
      <th>7208</th>
      <td>Eternal Sunshine of the Spotless Mind</td>
      <td>2004</td>
      <td>3758</td>
      <td>7</td>
      <td>12.9063</td>
      <td>7.806818</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>Cinema Paradiso</td>
      <td>1988</td>
      <td>834</td>
      <td>8</td>
      <td>14.177</td>
      <td>7.784420</td>
    </tr>
    <tr>
      <th>876</th>
      <td>Vertigo</td>
      <td>1958</td>
      <td>1162</td>
      <td>8</td>
      <td>18.2082</td>
      <td>7.711735</td>
    </tr>
    <tr>
      <th>4843</th>
      <td>Amélie</td>
      <td>2001</td>
      <td>3403</td>
      <td>7</td>
      <td>12.8794</td>
      <td>7.702024</td>
    </tr>
    <tr>
      <th>24982</th>
      <td>The Theory of Everything</td>
      <td>2014</td>
      <td>3403</td>
      <td>7</td>
      <td>11.853</td>
      <td>7.702024</td>
    </tr>
  </tbody>
</table>
</div>



# Content Based Recommender (Movie Description)

This part recommends movies that are similar to a particular movie in terms of movie description. Consider the pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score.


```python
movies['overview'].head(3)
```




    0    Led by Woody, Andy's toys live happily in his ...
    1    When siblings Judy and Peter discover an encha...
    2    A family wedding reignites the ancient feud be...
    Name: overview, dtype: object




```python
# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer Object. Remove all english stop words.
vect_1 = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
movies['overview'] = movies['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = vect_1.fit_transform(movies['overview'])

# Output the shape of tfidf_matrix
tfidf_matrix.shape
```




    (45466, 75827)



Use the cosine similarity to denote the similarity between two movies.


```python
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

Define a function that takes in a movie title as an input and outputs a list of the 8 most similar movies.


```python
#Construct a reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
```


```python
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):
    
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 8 most similar movies
    sim_scores = sim_scores[1:9]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 8 most similar movies
    return movies['title'].iloc[movie_indices]
```


```python
get_recommendations('X-Men')
```




    32251                              Superman
    20806                    Hulk vs. Wolverine
    23472    Mission: Impossible - Rogue Nation
    13635              X-Men Origins: Wolverine
    6195                                     X2
    30067                               Holiday
    21296                         The Wolverine
    38010                              Sharkman
    Name: title, dtype: object




```python
get_recommendations('Mission: Impossible - Ghost Protocol')
```




    23472         Mission: Impossible - Rogue Nation
    10952                    Mission: Impossible III
    3501                      Mission: Impossible II
    19275    The President's Man: A Line in the Sand
    26633                          A Dangerous Place
    18674                               Act of Valor
    15886                  My Girlfriend's Boyfriend
    33441                             Swat: Unit 887
    Name: title, dtype: object



### For the recommendations, it seems that the movies are correctly recommended based on similar movie descriptions. However, some users might like a movie based on the movie's cast, director and/or the genre of the movie. Hence, the model will be improved based on these two added features.


```python
# Load keywords and credits
credits = pd.read_csv("./movies-dataset/credits.csv")

# Remove rows with bad IDs.
movies = movies.drop([19730, 29503, 35587])

# Convert IDs to integers for merging
credits['id'] = credits['id'].astype('int')
movies['id'] = movies['id'].astype('int')
```


```python
# Merge credits into movies dataframe
movies = movies.merge(credits, on='id')
```


```python
movies.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.9469</td>
      <td>[{'name': 'Pixar Animation Studios', 'id': 3}]</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Toy Story</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>17.0155</td>
      <td>[{'name': 'TriStar Pictures', 'id': 559}, {'na...</td>
      <td>1995-12-15</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Jumanji</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>11.7129</td>
      <td>[{'name': 'Warner Bros.', 'id': 6194}, {'name'...</td>
      <td>1995-12-22</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Grumpier Old Men</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
      <td>[{'cast_id': 2, 'character': 'Max Goldman', 'c...</td>
      <td>[{'credit_id': '52fe466a9251416c75077a89', 'de...</td>
    </tr>
  </tbody>
</table>
</div>



From the merged dataframe, the scope of features will be defined as such:

**Crew:** Only the Director will be selected as I feel his directing sense contributes most to the movie.

**Cast:** Most movies have a mixture of better known and lesser known actors and actresses. Hence, I will choose only the top 3 actors/actresses names in the list.


```python
movies['cast'] = movies['cast'].apply(literal_eval)
movies['crew'] = movies['crew'].apply(literal_eval)
# movies['cast_size'] = movies['cast'].apply(lambda x: len(x))
# movies['crew_size'] = movies['crew'].apply(lambda x: len(x))
```


```python
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
```


```python
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
```


```python
# Define new director, cast and genres features that are in a suitable form.
movies['director'] = movies['crew'].apply(get_director)
```


```python
# features = ['cast', 'genres']
features = ['cast']
for feature in features:
    movies[feature] = movies[feature].apply(get_list)
```


```python
# Print the new features of the first 3 films
movies[['title', 'cast', 'director', 'genres']].head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>cast</th>
      <th>director</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story</td>
      <td>[Tom Hanks, Tim Allen, Don Rickles]</td>
      <td>John Lasseter</td>
      <td>[Animation, Comedy, Family]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jumanji</td>
      <td>[Robin Williams, Jonathan Hyde, Kirsten Dunst]</td>
      <td>Joe Johnston</td>
      <td>[Adventure, Fantasy, Family]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grumpier Old Men</td>
      <td>[Walter Matthau, Jack Lemmon, Ann-Margret]</td>
      <td>Howard Deutch</td>
      <td>[Romance, Comedy]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Function to convert all strings to lower case and strip names of spaces

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```


```python
print (movies['cast'].head(1).all())
```

    ['Tom Hanks', 'Tim Allen', 'Don Rickles']



```python
print (movies['director'].head())
```

    0      John Lasseter
    1       Joe Johnston
    2      Howard Deutch
    3    Forest Whitaker
    4      Charles Shyer
    Name: director, dtype: object



```python
print (movies['genres'].head())
```

    0     [Animation, Comedy, Family]
    1    [Adventure, Fantasy, Family]
    2               [Romance, Comedy]
    3        [Comedy, Drama, Romance]
    4                        [Comedy]
    Name: genres, dtype: object



```python
# Apply clean_data function to your features.
# features = ['cast', 'director', 'genres']
features = ['director', 'genres']

for feature in features:
    movies[feature] = movies[feature].apply(clean_data)
```


```python
def create_soup(x):
    return ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
```


```python
# Create a new soup feature
movies['soup'] = movies.apply(create_soup, axis=1)
```


```python
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

vect_2 = CountVectorizer(stop_words='english')
count_matrix = vect_2.fit_transform(movies['soup'])
```


```python
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
```


```python
# Reset index of your main DataFrame and construct reverse mapping as before
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'])
```


```python
get_recommendations('The Dark Knight Rises', cosine_sim2)
```




    12525      The Dark Knight
    10158        Batman Begins
    11399         The Prestige
    23964            Quicksand
    516      Romeo Is Bleeding
    8990        State of Grace
    11460          Harsh Times
    14977          Harry Brown
    Name: title, dtype: object




```python
get_recommendations('The Godfather', cosine_sim2)
```




    1187      The Godfather: Part II
    1922     The Godfather: Part III
    3996            Gardens of Stone
    3145            Scent of a Woman
    15503            The Rain People
    1174              Apocalypse Now
    1844           On the Waterfront
    5281                 The Gambler
    Name: title, dtype: object



# Collaborative Filter Recommender (User to User)


```python
# Load ratings.
ratings = pd.read_csv("./movies-dataset/ratings.csv")
ratings.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26024289 entries, 0 to 26024288
    Data columns (total 4 columns):
    userId       int64
    movieId      int64
    rating       float64
    timestamp    int64
    dtypes: float64(1), int64(3)
    memory usage: 794.2 MB



```python
# Load ratings.
ratings2 = pd.read_csv("./movies-dataset/ratings_small.csv")
ratings2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100004 entries, 0 to 100003
    Data columns (total 4 columns):
    userId       100004 non-null int64
    movieId      100004 non-null int64
    rating       100004 non-null float64
    timestamp    100004 non-null int64
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB



```python
small_data = ratings.sample(frac=0.1)
small_data.sort_values(by=['userId'])
print(small_data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2602429 entries, 18281416 to 22564926
    Data columns (total 3 columns):
    userId     int64
    movieId    int64
    rating     float64
    dtypes: float64(1), int64(2)
    memory usage: 79.4 MB
    None



```python
small_data.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18281416</th>
      <td>189649</td>
      <td>1196</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>2993381</th>
      <td>31254</td>
      <td>47</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9133599</th>
      <td>94302</td>
      <td>5377</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>110</td>
      <td>1.0</td>
      <td>1425941529</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>147</td>
      <td>4.5</td>
      <td>1425942435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>858</td>
      <td>5.0</td>
      <td>1425941523</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings2.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert IDs to integers for merging
credits['id'] = credits['id'].astype('int')
movies['id'] = movies['id'].astype('int')
```


```python
ratings = ratings.drop(['timestamp'], axis=1)
```


```python
ratings2 = ratings2.drop(['timestamp'], axis=1)
```


```python
ratings.shape
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-31-16e16fd98d55> in <module>()
    ----> 1 ratings.shape
    

    NameError: name 'ratings' is not defined



```python
ratings2.shape
```




    (100004, 3)




```python
# Number of unique users
len(set(ratings.userId))
```




    270896




```python
# Number of unique users
len(set(ratings2.userId))
```




    671




```python
# Number of unique users
len(set(small_data.userId))
```




    229823




```python
ratings2_pivot = ratings2.pivot('userId','movieId','rating')

ratings2_pivot
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>161084</th>
      <th>161155</th>
      <th>161594</th>
      <th>161830</th>
      <th>161918</th>
      <th>161944</th>
      <th>162376</th>
      <th>162542</th>
      <th>162672</th>
      <th>163949</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>642</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>643</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>644</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>645</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>646</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>647</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>648</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>649</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>650</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>651</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>652</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>653</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>654</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>655</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>656</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>657</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>658</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>659</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>660</th>
      <td>2.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>661</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>662</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>663</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>664</th>
      <td>3.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>665</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>666</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>667</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>668</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>669</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>670</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>671</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>671 rows × 9066 columns</p>
</div>




```python
def jaccard(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return Inf   
    # & = interesection
    # | = union
    return (float(len(set1 & set2)) / len(set1 | set2))
```


```python
jaccard(ratings['userId']==1,ratings['userId']==99)
```




    1.0



### Weighted Jaccard

This metric does not fully capture our intution of distance between two users and the movies they watch. Let's add a weighting which emphasizes movies with higher vote counts.

Weight each movie movie's total vote. This is a useful measure, since we want a large weight with 1000 votes and a much smaller weight with 2 votes.


```python
def weighted_jaccard(set1, set2):
    return (float(movies['vote_count']['x'] for movie in set1 & set2)) /
            sum(brand_freq.get(brand) for brand in set1 | set2))
```

### Below is the part where SVD is implemented


```python
reader = Reader()
```


```python
ratings3 = pd.read_csv("./movies-dataset/ratings_small.csv")
ratings3.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = Dataset.load_from_df(ratings3[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
```


```python
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
```

    Evaluating RMSE, MAE of algorithm SVD.
    
    ------------
    Fold 1
    RMSE: 0.8947
    MAE:  0.6894
    ------------
    Fold 2
    RMSE: 0.8995
    MAE:  0.6926
    ------------
    Fold 3
    RMSE: 0.8910
    MAE:  0.6868
    ------------
    Fold 4
    RMSE: 0.8996
    MAE:  0.6917
    ------------
    Fold 5
    RMSE: 0.8991
    MAE:  0.6929
    ------------
    ------------
    Mean RMSE: 0.8968
    Mean MAE : 0.6907
    ------------
    ------------





    CaseInsensitiveDefaultDict(list,
                               {'mae': [0.68937359473806903,
                                 0.69259939130111503,
                                 0.68678665677980999,
                                 0.69169120460418154,
                                 0.69285620150031413],
                                'rmse': [0.89472414482943841,
                                 0.89948598218998499,
                                 0.89096153777913623,
                                 0.8996171912501465,
                                 0.89907130432515781]})




```python
trainset = data.build_full_trainset()
svd.train(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x11704a908>




```python
ratings3[ratings3['userId'] == 1]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1263</td>
      <td>2.0</td>
      <td>1260759151</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1287</td>
      <td>2.0</td>
      <td>1260759187</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1293</td>
      <td>2.0</td>
      <td>1260759148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1339</td>
      <td>3.5</td>
      <td>1260759125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1343</td>
      <td>2.0</td>
      <td>1260759131</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1371</td>
      <td>2.5</td>
      <td>1260759135</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1405</td>
      <td>1.0</td>
      <td>1260759203</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1953</td>
      <td>4.0</td>
      <td>1260759191</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2105</td>
      <td>4.0</td>
      <td>1260759139</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2150</td>
      <td>3.0</td>
      <td>1260759194</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2193</td>
      <td>2.0</td>
      <td>1260759198</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2294</td>
      <td>2.0</td>
      <td>1260759108</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2455</td>
      <td>2.5</td>
      <td>1260759113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2968</td>
      <td>1.0</td>
      <td>1260759200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3671</td>
      <td>3.0</td>
      <td>1260759117</td>
    </tr>
  </tbody>
</table>
</div>




```python
svd.predict(1, 31)
```




    Prediction(uid=1, iid=31, r_ui=None, est=2.5823946941598028, details={'was_impossible': False})


