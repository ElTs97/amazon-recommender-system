# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

# Load the dataset containing Amazon product reviews
df = pd.read_csv('amazon_reviews.csv', header=None)
df.columns = ['userid', 'productid', 'rating', 'timestamp']
df.head()

# View basic dataset information such as data types and non-null counts
print(df.info())

# Check the shape of the dataset (number of rows and columns)
print(df.shape)

# Display basic statistics for the numerical columns ('rating', etc.)
print(df.describe())

# Check for missing values in the dataset
print(df.isnull().sum())

# Identify and count duplicate rows in the dataset
duplicate_rows = df.duplicated()
print(f'Total duplicates: {duplicate_rows.sum()}')

# Visualize the distribution of ratings (1-5) across the dataset
plt.figure(figsize=(10,6))
sns.histplot(df['rating'], bins=5, kde=False)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Find the top 15 users who have given the most ratings

# Group by 'userid' and count the number of ratings per user
user_rating_counts = df.groupby('userid').size().reset_index(name='rating_count')

# Sort by 'rating_count' in descending order to find the most active users
top_users = user_rating_counts.sort_values(by='rating_count', ascending=False).head(15)
print(top_users)

# Analyze product ratings

# Group by 'productid' and calculate both the count of ratings and the average rating for each product
product_rating_summary = df.groupby('productid').agg(
    rating_count=('rating', 'size'),        # Number of ratings
    average_rating=('rating', 'mean')       # Average rating
).reset_index()

# Sort products by the number of ratings in descending order
product_rating_summary = product_rating_summary.sort_values(by='rating_count', ascending=False)
print(product_rating_summary.head(15))

# Create an interactive scatter plot to visualize the relationship between the number of ratings and average rating for each product
fig = px.scatter(
    product_rating_summary,
    x='average_rating',
    y='rating_count',
    title='Number of Ratings vs Average Rating by Product',
    labels={'rating_count': 'Number of Ratings', 'average_rating': 'Average Rating'},
    hover_data=['productid'],  
)
fig.update_xaxes(type='log')  # Log scale for better visualization of the spread
fig.show()

# ---- Part 1: Product recommendation based on similar products ----

# Take a subset of the dataset for further analysis (first 10,000 rows)
df_sub = df.head(10000)

# Create a user-item utility matrix where rows represent users and columns represent products (ratings as values)
util_matrix = df_sub.pivot_table(values='rating', index='userid', columns='productid', fill_value=0)
util_matrix.head()

# Get the shape of the utility matrix (number of users and products)
util_matrix.shape

# Transpose the utility matrix to get a product-user matrix
transposed_matrix = util_matrix.T
transposed_matrix.head()

# Check the shape of the transposed matrix (number of products, number of users)
transposed_matrix.shape

# Apply Truncated SVD to reduce dimensionality (reduce the product-user matrix to 10 latent factors)
SVD = TruncatedSVD(n_components=10)
decomposed_mat = SVD.fit_transform(transposed_matrix)
decomposed_mat.shape

# Compute the correlation matrix between products based on the latent factors
corr_mat = np.corrcoef(decomposed_mat)
corr_mat.shape

# Select a product by index (e.g., the 100th product) to find similar items
transposed_matrix.index[100]

# Assign a product ID manually for demonstration purposes (this could be dynamic in a real system)
i = '1616833742'
product_names = list(transposed_matrix.index)
product_ID = product_names.index(i)
product_ID

# Find correlations of all products with the selected product based on customer ratings
corr_product_ID = corr_mat[product_ID]
corr_product_ID.shape

# Recommend top 5 highly correlated products (similar products based on user behavior)

# Filter products with a correlation score > 0.90, indicating high similarity
Recommend = list(transposed_matrix.index[corr_product_ID > 0.90])

# Remove the product that the user has already purchased
Recommend.remove(i)

# Display the top 5 recommended products
Recommend[0:5] 

# ---- Part 2: Product recommendation based on textual clustering ----

# Load the product descriptions dataset
product_desc = pd.read_csv('Amazon.csv')
product_desc.head()
product_desc.info()
product_desc.shape

# Check for missing values in the dataset
print(product_desc.isnull().sum())

# Drop rows with missing values
product_desc = product_desc.dropna()

# Keep only relevant columns for product description
cols = ['product_id', 'about_product']
product_desc = product_desc[cols]
product_desc.head()

# Function to clean the text by removing non-alphabet characters and converting to lowercase
def clean(text):
    res = re.sub("[^A-Za-z]", " ", text)
    res = res.strip().lower()
    return res

# Apply the cleaning function to the 'about_product' column
product_desc["about_product"] = product_desc["about_product"].apply(clean)
product_desc.head()

# Select a subset of product descriptions for further analysis
product_desc_fe = product_desc.head(1000)
product_desc_fe["about_product"].head(10)

# Feature extraction: Convert the cleaned product descriptions into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_desc_fe["about_product"])
X1

# Visualize the product clusters

# Set X as the TF-IDF matrix
X = X1

# Apply KMeans clustering to group similar products into 10 clusters
kmeans = KMeans(n_clusters=10, init= 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Plot the clusters to visualize how products are grouped
plt.plot(y_kmeans, ".")
plt.show()

# Function to print the top terms in each cluster
def print_cluster(i):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

# Re-run the clustering with optimal parameters
true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

# Print the top terms associated with each cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(true_k):
    print_cluster(i)

# Function to predict the cluster for a given product description and print the top terms for that cluster
def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    print_cluster(prediction[0])

# Test the recommendation system with a sample product description (e.g., "supports core")
show_recommendations("supports core")
