import torch
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

# # loaded saved features and ids
# text_features = torch.load('text_features.pt')
# image_features = torch.load('image_features.pt')
# posting_ids = torch.load('posting_ids.pt')
# concat_features = np.concatenate((image_features, image_features), axis=1)

# # Use KNN for finding similar products
# knn_model = NearestNeighbors(n_neighbors=2, metric="cosine")
# knn_model.fit(concat_features)

# similar_products = {}
# for i, feature in enumerate(concat_features):
#     distances, indices = knn_model.kneighbors([feature])
#     similar_postings = [posting_ids[idx] for idx in indices.flatten() if idx != i]
#     similar_products[posting_ids[i]] = similar_postings

def combine_matches(image_match_file, text_match_file):
    # load image matches
    with open(image_match_file, 'rb') as f:
        image_matches = pickle.load(f)

    # load text matches
    with open(text_match_file, 'rb') as f:
        text_matches = pickle.load(f)

    # combine
    combined_matches = {}
    for product, img_similars in image_matches.items():
        txt_similars = text_matches.get(product, [])
        combined_matches[product] = set(img_similars) | set(txt_similars)  # take the union

    return combined_matches 

combined_matches = combine_matches("knn_image_matches.pkl", "text_matches.pkl")
# print("combined_matches:", combined_matches)

df = pd.read_csv("train_subset.csv")

# mapping true similar pairs according label_mapping
true_similar_pairs = set()
for label, items in df.groupby("label_group")["posting_id"]:
    item_pairs = {(i, j) for i in items for j in items if i != j}
    true_similar_pairs.update(item_pairs)

# # Initialize a dictionary
# posting_id_pair_counts = defaultdict(set)

# # Record similar products for each posting_id
# for item1, item2 in true_similar_pairs:
#     posting_id_pair_counts[item1].add(item2)
#     posting_id_pair_counts[item2].add(item1)

# # Calculate the average number of similar products for each posting_id
# average_similar_count = sum(len(items) for items in posting_id_pair_counts.values()) / len(posting_id_pair_counts)

# # Print
# print("The number of similar products for each posting_id:")
# for posting_id, items in posting_id_pair_counts.items():
#     print(f"{posting_id}: {len(items)}")

# print(f"\nAverage number of similar products for each posting_id: {average_similar_count:.2f}")


# Convert `combined_matches` into a set of product pairs
predicted_similar_pairs = set()
for product, similars in combined_matches.items():
    for similar_product in similars:
        # Avoid duplicate pairs (a, b) and (b, a) in different orders
        pair = tuple(sorted((product, similar_product)))
        predicted_similar_pairs.add(pair)


# predicted_posting_id_pair_counts = defaultdict(set)

# for item1, item2 in predicted_similar_pairs:
#     predicted_posting_id_pair_counts[item1].add(item2)
#     predicted_posting_id_pair_counts[item2].add(item1)

# predicted_average_similar_count = sum(len(items) for items in predicted_posting_id_pair_counts.values()) / len(predicted_posting_id_pair_counts)

# # print("The number of predicted similar products for each posting_id:")
# for posting_id, items in predicted_posting_id_pair_counts.items():
#     print(f"{posting_id}: {len(items)}")

# # print(f"\nAverage number of predicted similar products for each posting_id: {predicted_average_similar_count:.2f}")

# Calculate the size of the intersection, i.e., the number of correct predictions
correct_predictions = predicted_similar_pairs & true_similar_pairs
num_correct_predictions = len(correct_predictions)
# print("correct_predictions:", correct_predictions)

## Evaluation
accuracy = num_correct_predictions / len(predicted_similar_pairs) if predicted_similar_pairs else 0

recall = num_correct_predictions / len(true_similar_pairs) if true_similar_pairs else 0

f1_score = (2 * accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")