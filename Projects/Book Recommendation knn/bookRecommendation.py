import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# get data files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT_DIR, 'book-crossings')

books_filename = os.path.join(PATH, 'BX-Books.csv')
ratings_filename = os.path.join(PATH, 'BX-Book-Ratings.csv')

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'}
)

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'}
)


# removing books with less than 100 ratings
tempDf = df_ratings.groupby(["isbn"]).count().reset_index()
booksWithMoreThan100Ratings = tempDf.loc[tempDf["rating"] >= 100]["isbn"]
booksWithMoreThan100Ratings = df_books.loc[df_books["isbn"].isin(booksWithMoreThan100Ratings)]

# removing users with less than 200 ratings
tempDf = df_ratings[["user", "rating"]].groupby(["user"]).count().reset_index()
usersWithMoreThan200Ratings = tempDf.loc[tempDf["rating"] >= 200]["user"]
df = df_ratings.loc[df_ratings["user"].isin(usersWithMoreThan200Ratings)]
df = df.loc[df["isbn"].isin(booksWithMoreThan100Ratings["isbn"])]
df = df.reset_index()

df.to_csv(os.path.join(PATH, 'cleanedData.csv'), encoding='utf-8', index=False) # creates a .csv file containing the cleaned data

# pivot ratings into book features
df_book_features = df.pivot(
    index='isbn',
    columns='user',
    values='rating'
).fillna(0)

# convert dataframe of book features to scipy sparse matrix
matrix_book_features = csr_matrix(df_book_features.values)

# create model
model = NearestNeighbors(metric="cosine")
model.fit(matrix_book_features)

def get_recommends(title=""):
    try:
        book = booksWithMoreThan100Ratings.loc[booksWithMoreThan100Ratings["title"] == title]
    except:
        print(f"Book {title} cannot be found")
        return [""]

    bookFeature = df_book_features.loc[df_book_features.index.isin(book["isbn"])]
    distance, indice = model.kneighbors([i for i in bookFeature.values], n_neighbors=6)

    distance =distance[0][1:]
    indice = indice[0][1:]

    titles = [df_books.loc[df_books['isbn'] == df_book_features.iloc[i].name]["title"].values[0] for i in indice]

    recommended = [list(i) for i in zip(titles, distance)][::-1]

    recommended_books = [title, recommended]

    return recommended_books

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
    for i in range(2):
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You havn't passed yet. Keep trying!")

test_book_recommendation()