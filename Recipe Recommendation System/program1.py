import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# Function to load and preprocess dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    recipe_ids = [recipe['id'] for recipe in data]
    ingredients = [' '.join(recipe['ingredients']) for recipe in data]
    ratings = [recipe.get('rating', 0.0) for recipe in data]  # Assuming 'rating' field in dataset
    return recipe_ids, ingredients, ratings


# Load training dataset and print head
train_ids, train_ingredients, train_ratings = load_dataset('train.json')
print(f"Head of training dataset (train.json):\n{list(zip(train_ids[:5], train_ingredients[:5], train_ratings[:5]))}\n")

# Load test dataset and print head
test_ids, test_ingredients, test_ratings = load_dataset('test.json')
print(f"Head of test dataset (test.json):\n{list(zip(test_ids[:5], test_ingredients[:5], test_ratings[:5]))}\n")

# Combine both datasets for TF-IDF vectorization
all_recipe_ids = train_ids + test_ids
all_ingredients = train_ingredients + test_ingredients

# TF-IDF vectorization on all ingredients
vectorizer = TfidfVectorizer(stop_words='english')
ingredients_matrix = vectorizer.fit_transform(all_ingredients)

# Reduce dimensionality using Truncated SVD (similar to PCA but for sparse matrices)
svd = TruncatedSVD(n_components=100)
ingredients_matrix_svd = svd.fit_transform(ingredients_matrix)

# Convert to sparse matrix for cosine similarity computation
ingredients_sparse = csr_matrix(ingredients_matrix_svd)


# Function to compute cosine similarity for a given batch
def compute_cosine_similarity(X1, X2):
    similarity_matrix = cosine_similarity(X1, X2)
    return similarity_matrix


# Collaborative Filtering: Recommend recipes based on user ratings similarity
def collaborative_filtering_recommendations(user_id, top_n=5):
    user_ratings_vec = np.array(train_ratings + test_ratings)
    sim_scores = cosine_similarity([user_ratings_vec], [user_ratings_vec])[0]

    # Sort indices based on similarity scores
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    recommended_recipes = [all_recipe_ids[idx] for idx, _ in sim_scores]
    return recommended_recipes


# Content-Based Filtering: Recommend recipes based on ingredients similarity
def content_based_recommendations(recipe_id, top_n=5):
    try:
        recipe_index = all_recipe_ids.index(recipe_id)
    except ValueError:
        print(f"Recipe ID {recipe_id} not found in dataset.")
        return []

    # Compute similarity only for the specific recipe
    similarity_scores = cosine_similarity([ingredients_matrix_svd[recipe_index]], ingredients_matrix_svd)[0]
    similar_recipe_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]

    recommended_recipes = [all_recipe_ids[idx] for idx in similar_recipe_indices]
    return recommended_recipes


# Hybrid Recommendations: Combine CF and CBF
def hybrid_recommendations(user_id, recipe_id, top_n=5):
    cf_recs = collaborative_filtering_recommendations(user_id, top_n=top_n)
    cbf_recs = content_based_recommendations(recipe_id, top_n=top_n)

    # Combine recommendations and remove duplicates
    combined_recs = list(set(cf_recs + cbf_recs))
    return combined_recs[:top_n]


# Evaluation Metrics: Example function to evaluate recommendation performance
def evaluate_recommendations():
    # Placeholder for evaluation metrics implementation (e.g., accuracy, precision)

    # Example: Print evaluation results
    print("Evaluation metrics for recommendation system:")
    print("-" * 50)
    print("To be implemented.")


# Main execution
if __name__ == "__main__":
    # Example: Recommend recipes based on user's preferences and dietary restrictions
    user_id = 1
    test_recipe_id = 18009

    print(f"Collaborative Filtering Recommendations for User {user_id}:")
    cf_recommendations = collaborative_filtering_recommendations(user_id)
    print(cf_recommendations)

    print(f"\nContent-Based Filtering Recommendations for Recipe ID {test_recipe_id}:")
    cbf_recommendations = content_based_recommendations(test_recipe_id)
    print(cbf_recommendations)

    print(f"\nHybrid Recommendations for User {user_id} and Recipe ID {test_recipe_id} (CF + CBF):")
    hybrid_rec = hybrid_recommendations(user_id, test_recipe_id)
    print(hybrid_rec)

    # Evaluate recommendation system performance
    evaluate_recommendations()
