import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
books = pd.read_csv('Books.csv', low_memory=False)
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

# Merge ratings with book details
ratings_with_name = ratings.merge(books, on='ISBN')

# Calculate the number of ratings and average rating for each book
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

# Combine the number of ratings and average rating into one DataFrame
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)

# Merge with books to get additional details including 'Book-Author' and 'Image-URL-M'
result = popular_df.merge(books[['Book-Title', 'Book-Author', 'Image-URL-M']], on='Book-Title').drop_duplicates('Book-Title')

# Print columns of the result DataFrame to confirm 'Book-Author' and 'Image-URL-M' exist
print("Result DataFrame columns after merge:", result.columns)

# Save the result DataFrame with all required columns
result[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']].to_pickle('popular.pkl')

print(result['Image-URL-M'].iloc[0])

# Model 2
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
final_ratings = final_ratings.drop_duplicates()  # Ensure duplicates are dropped

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)

# Save the required objects
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))
