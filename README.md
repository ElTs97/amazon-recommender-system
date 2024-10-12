Amazon Product Recommendation System

This project implements a product recommendation system based on two different approaches: collaborative filtering and text clustering. The goal is to recommend products to users based on their purchase history and product descriptions.

Project Structure
The project is divided into two main parts:

1. Product Recommendation Based on Similar Products (Collaborative Filtering)
2. Product Recommendation Based on Product Descriptions (Text Clustering)

Project Overview
This repository contains Python and Jupyter(to show how the systems work) code to build two types of recommendation systems:

1. Collaborative Filtering: Recommends products based on the behavior of users who have similar purchasing patterns.
2. Text Clustering: Recommends products based on similarities in product descriptions using unsupervised text clustering.

Dataset
Amazon Reviews Dataset: This dataset contains product reviews from users, including:

- userid: ID of the user who rated the product
- productid: ID of the product being rated
- rating: User's rating for the product (1–5)
- timestamp: Time when the rating was given

Amazon Product Descriptions Dataset: This dataset contains the product descriptions used for text clustering:

- product_id: Unique ID for the product
- about_product: Text description of the product

Approaches
1. Collaborative Filtering (Based on Similar Products)
In this method, we recommend products to users by analyzing product ratings given by other users who have purchased similar items.

Steps:
- Utility Matrix: We create a user-item utility matrix where each row represents a user, and each column represents a product. The matrix is filled with user ratings.
- Matrix Factorization (SVD): We use Truncated Singular Value Decomposition (SVD) to reduce the dimensionality of the matrix and find latent factors.
- Correlation Matrix: Based on these factors, a correlation matrix is generated to measure the similarity between different products.
- Product Recommendation: If a user has purchased a particular product, we recommend other products with a high correlation score (similarity) to that product.

2. Text Clustering (Based on Product Descriptions)
This method clusters products based on the textual similarity of their descriptions, allowing us to recommend products that are similar in terms of their content.

Steps:
- Text Preprocessing: The product descriptions are cleaned (removing special characters, converting to lowercase, etc.).
- TF-IDF Vectorization: Text features are extracted from the descriptions using the TF-IDF (Term Frequency-Inverse Document Frequency) method.
- KMeans Clustering: The products are clustered into groups based on their textual descriptions using the KMeans algorithm.
- Cluster Analysis: For a given product, we recommend other products in the same cluster, as they are likely to share similar features.

Usage
You can use the recommendation system for different use cases:

1. Collaborative Filtering: If you have user purchase history and ratings, use the collaborative filtering method to recommend products that other users with similar tastes have purchased.

- To recommend products based on a product a user purchased, modify the product ID in the code.
2. Text Clustering: If you have product descriptions and want to recommend products with similar features or attributes, use the text clustering approach. You can input keywords or product descriptions to get relevant recommendations.

- Modify the input to the show_recommendations() function to test it with different products.