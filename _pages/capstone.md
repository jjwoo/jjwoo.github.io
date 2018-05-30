---
layout: single
permalink: capstone/
title: &title "WatchThis (A Movie Recommender)"
author_profile: true
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
<img src="/assets/images/movie-collage.jpg" width="100%">

## WatchThis (A Movie Recommender)

## 1. Problem Statement

Have you ever wanted to watch a movie in the comfort of your home, but stopped short of doing so just because nothing came to mind? Or during a stayover with friends, too many differences in preferences for a movie choice led to a wild goose catch for a decision that never materialized? 

Yes, there are a few movie recommender resources around. They are either complicated to use (ie. need to set up account, enter a bunch of user preferences etc), or too simple and does not allow for a flexibility in preferences for input.

## 2. Natural Language Processing (NLP)

This project is heavily reliant on Natural Language Processing or NLP, so let us understand what NLP is all about.

NLP is broadly defined as the processing or manipulation of natural language, which can be in the form of speech and text etc. It is a very challenging task to make "useful" sense of such information, as they are messy and can be illogical at times. 

For the purpose of this project, we will be comparing how "similar" the movie plots, cast, director are, and subsequently recommend the movies accordingly. More details will be discussed as we go along.

## 3. Dataset

The dataset is from GroupLens Research 20M. It is a stable benchmark dataset.

```python
# Load the data.
movies = pd.read_csv("./movies-dataset/movies_metadata.csv")
movies.head(20)
```

<iframe src="/assets/capstone-table1.html" height="400" width="600" overflow="auto"></iframe>

After some data cleaning, we are good to go!


## 4. Baseline Model: Simple Recommender

The Simple Recommender is a baseline model which offers simple recommendations based on a movie's weighted rating. Then, the top few selected movies will be displayed. 

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
    

I will use a weighted rating that takes into account the average rating and the number of votes for the movie. Hence, a movie that has a 8 rating from say, 50k voters will have a higher score than a movie with the same rating but with fewer voters.

<img src="/assets/images/weightrating.jpg" width="10%">

where,
* *v* is the number of votes for the movie
* *m* is the minimum votes required to be listed in the chart
* *R* is the average rating of the movie

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
```


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

## 4. Content Based Recommender (Movie Description)

This part recommends movies that are similar to a particular movie in terms of movie description. It considers the pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score.


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



### 4.1 Content Based Recommender (Other Parameters)

For the recommendations, it seems that the movies are correctly recommended based on similar movie descriptions. However, some users might like a movie based on the movie's cast, director and/or the genre of the movie. Hence, the model will be improved based on these two added features.


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




## 5. Prediction Of Ratings (Collaborative Filtering)

For this part, I will attempt to predict how a user will rate a recommended movie (presuming he or she has not seen it before or at least has not rated it before)

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
    
    
## 6. Key Insights

1. **Baseline Model**
    - Does well in recommending movies which have a high weighted rating according to user's favourite genre.
    - Not flexible enough to take in more parameters and recommend more personalized choices for user.  

2. **Content Based Model**
    - Does well in recommending movies which are similar to the user's inputs, such as movie plot, favourite director etc.
    - Does not have cold start problem as user does not need to have rated many movies before, since the model just needs user to select favourite movie and other parameters if he/she so wishes.  
    
3. **Rating Prediction Model**   
    - The Surprise package, which is a Python scikit package for recommender systems, has a decent performance.
    - Does not address the cold start problem, which occurs when the user has not rated enough movies before.  
    
4. **Other models**
    For the movie recommender engine, there exists a Collaborative Filtering model which takes into account similar users' choices of movies and recommends such movies to the inquiring user. But due to the time constraints of the capstone project, we are only able to explore content based model. I will be following up with this model so give this a space a watch!

## 7. Future work

I hope you like what you have seen thus far.

If you have any comments or questions regarding the above work, feel free to contact me via the "Contact Me" tab at the top of the page.

Have a nice day!
