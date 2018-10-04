import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
from sklearn.decomposition import TruncatedSVD

# Constants
FONT_SIZE = 10
data_path = './data/'
BAR_CHART_MODEL = 'bar'
# end of constants

books = pd.read_csv(data_path + 'BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                 'imageUrlL']
users = pd.read_csv(data_path + 'BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv(data_path + 'BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

"""
Ratings data
"""
print("Following are ratings data\n")
print(ratings.shape)
print(list(ratings.columns))
print("----------------------------\n\n")

"""
The ratings are very unevenly distributed 
and according to the following plot
the vast majority of ratings are 0
"""
print("Ratings distribution plot opened ... ")
plt.rc("font", size=FONT_SIZE)
ratings.bookRating.value_counts(sort=False).plot(kind=BAR_CHART_MODEL)
plt.title("Rating distribution\n")
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()

print("Ratings distribution plot has been closed.")
print("----------------------------")
# end of plot


"""
Books data
"""
print("Following are books data\n")
print(books.shape)
print(list(books.columns))
print("----------------------------")

"""
Users data
"""
print("Following are users data\n")
print(books.shape)
print(list(books.columns))
print("----------------------------")

"""
The most active users are among those in their 20–30s.
"""
print("Distribution plot opened ... ")
users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('system2.png', bbox_inches='tight')
plt.show()

print("Distribution plot has been closed.")
print("----------------------------")

# end of plot

"""
Recommendations based on rating counts
"""
print("\n\n")
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()

# Test data
most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'],
                                index=np.arange(5), columns=['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
print(most_rated_books_summary)

average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
print(average_rating.sort_values('ratingCount', ascending=False).head())

counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]

counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

"""
We convert the ratings table to a 2D matrix. 
The matrix will be sparse because
not every user rated every book.
"""
ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
print(ratings_pivot.shape)
print(ratings_pivot.head())

# Test data
bones_ratings = ratings_pivot['0316666343']
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
corr_summary[corr_summary['ratingCount'] >= 300].sort_values('pearsonR', ascending=False).head(10)

books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972', '0684872153'],
                                  index=np.arange(9), columns=['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
print(corr_books)

# Collaborative Filtering Using k-Nearest Neighbors (kNN)
combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
print(combine_book_rating.head())

"""
Group by book titles and create a new column
for total rating count.
"""
combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])

book_ratingCount = (combine_book_rating.
    groupby(by=['bookTitle'])['bookRating'].
    count().
    reset_index().
    rename(columns={'bookRating': 'totalRatingCount'})
[['bookTitle', 'totalRatingCount']]
    )
print(book_ratingCount.head())

"""
Combine the rating data with the total rating count data, 
this gives us exactly what we need to find out
which books are popular and filter out
lesser-known books.
"""
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle',
                                                         how='left')
print(rating_with_totalRatingCount.head())

"""
Statistics of total rating count
"""
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['totalRatingCount'].describe())

print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
print(rating_popular_book.head())
combined = rating_popular_book.merge(users, left_on='userID', right_on='userID', how='left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
print(us_canada_user_rating.head())

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID',
                                                          values='bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(us_canada_user_rating_matrix)
NearestNeighbors(algorithm='brute', leaf_size=30, metric='cosine',
                 metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),
                                          n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]],
                                                       distances.flatten()[i]))

us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index='userID', columns='bookTitle',
                                                           values='bookRating').fillna(0)
print(us_canada_user_rating_pivot2.head())


X = us_canada_user_rating_pivot2.values.T

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)

warnings.filterwarnings("ignore", category=RuntimeWarning)
corr = np.corrcoef(matrix)

us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)

coffey_hands = us_canada_book_list.index("The Killing Game: Only One Can Win...and the Loser Dies")
print(coffey_hands)

corr_coffey_hands = corr[coffey_hands]

print(list(us_canada_book_title[(corr_coffey_hands <= 0.9)]))
