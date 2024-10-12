import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('amazon_reviews.csv', header=None)
df.columns=['userid', 'productid', 'rating', 'timestamp']
df.head()

# view basic information
print(df.info())
print(df.shape)

# display basic statistics
print(df.describe())

# check for missing values
print(df.isnull().sum())

# check for duplicates
duplicate_rows = df.duplicated()
print(f'Total duplicates: {duplicate_rows.sum()}')

# distribution of ratings
plt.figure(figsize=(10,6))
sns.histplot(df['rating'], bins = 5)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Top 15 users with most ratings

# Group by 'userid' and count the number of ratings
user_rating_counts = df.groupby('userid').size().reset_index(name='rating_count')

# Sort by the number of ratings in descending order and print the top 15
top_users = user_rating_counts.sort_values(by='rating_count', ascending=False).head(15)
print(top_users)

# Group by 'product_id' and aggregate
product_rating_summary = df.groupby('productid').agg(
    rating_count=('rating', 'size'),        # Count of ratings
    average_rating=('rating', 'mean')       # Average rating
).reset_index()

# Sort by the number of ratings in descending order
product_rating_summary = product_rating_summary.sort_values(by='rating_count', ascending=False)
print(product_rating_summary.head(15)) 

# Create an interactive scatter plot
fig = px.scatter(
    product_rating_summary,
    x='average_rating',
    y='rating_count',
    title='Number of Ratings vs Average Rating by Product',
    labels={'rating_count': 'Number of Ratings', 'average_rating': 'Average Rating'},
    hover_data=['productid'],  
)
fig.update_xaxes(type='log')
fig.show()