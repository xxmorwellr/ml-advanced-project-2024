import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


# Load data
df = pd.read_csv("train_subset.csv")

# Text preprocessing
print("Text preprocessing...")
def clean_title(title):
    # 1. Process by replacing and handling escape characters
    # First, convert escape sequences in the string to bytes
    bytes_title = bytes(title, 'latin1').decode('unicode_escape')
    # Then, decode bytes into a UTF-8 string
    title = bytes_title.encode('latin1').decode('utf-8')
    # 2. Normal cleaning
    # title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)  # Keep only letters and numbers?
    title = title.lower()  # Unify lower case
    return title

# Apply
df['cleaned_title'] = df['title'].apply(clean_title)
# print(df[['title', 'cleaned_title']].head(20))

# 1. Using tf-idf to extract KEY WORDS
print("TF-IDF Key Words Extracting...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.8) ## Adjusting parameters..
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_title'])

# Extract key words and corresponding tf-idf values
feature_names = tfidf_vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
df_tfidf = pd.DataFrame(dense, columns=feature_names)

# # Filter features
# threshold = 0.5 
# df_filtered = df_tfidf.loc[:, (df_tfidf.max(axis=0) > threshold)]

# # print
# print("TF-IDF Filtered Result:")
# print(df_tfidf)

# # look up specific document
# # doc_index = 4
# doc_index = 38  
# tfidf_values = df_tfidf.iloc[doc_index]
# print("tfidf_values:", tfidf_values)

# # find out the key word with max tf-idf value
# max_tfidf_value = tfidf_values.max()
# max_tfidf_keyword = tfidf_values.idxmax()

# # find out key words with top_2 max tf-idf values
# top_n = 2  # adjustable
# top_tfidf_keywords = tfidf_values.nlargest(top_n)

# print(f"The key word with max tf-idf value in doc {doc_index + 1}: '{max_tfidf_keyword}'，value: {max_tfidf_value:.4f}")
# print(f"The key words with top 2 max tf-idf value in doc {doc_index + 1}: '{top_tfidf_keywords}, value: {top_tfidf_keywords:.4f}'")


# 2. Using BERT to extract text features
# Load BERT model and tokenizer
print("BERT Text Features Extracting...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define extract funtion
def extract_bert_features(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] token
    return outputs.last_hidden_state[:, 0, :].numpy()

# Extact BERT features for each title
bert_features = [extract_bert_features(title) for title in df['cleaned_title']]
bert_features_array = torch.vstack([torch.tensor(feature) for feature in bert_features]).numpy()

# Save for further cocatenation
# torch.save(bert_features_array, 'text_features.pt')


## Calculate similarities
# 1. Use TF-IDF to calculate similarity (coarse filtering)
# Calculate the TF-IDF similarity between each pair of products
print("Filtering out preliminarily relevant product pairs...")
tfidf_sim_matrix = cosine_similarity(df_tfidf)
# print("tfidf_sim_matrix:", tfidf_sim_matrix)
# Set a similarity threshold to filter out preliminarily relevant product pairs
tfidf_sim_threshold = 0.3
tfidf_pairs = np.argwhere(tfidf_sim_matrix > tfidf_sim_threshold)

# 2. Calculate similarity using BERT (fine filtering)
# Calculate cosine similarity between BERT feature vectors
print("Filtering more accurate product pairs...")
bert_sim_matrix = cosine_similarity(bert_features_array)
# print("bert_sim_matrix:", bert_sim_matrix)
# Set BERT similarity threshold to filter more accurate product pairs
bert_sim_threshold = 0.8
bert_pairs = np.argwhere(bert_sim_matrix > bert_sim_threshold)

# 3. Combine (Intersection)
print("Combining similar product pairs...")
similar_pairs = set(map(tuple, tfidf_pairs)) & set(map(tuple, bert_pairs))

text_matches = {}
for product1, product2 in similar_pairs:
    text_matches.setdefault(product1, set()).add(product2)
    text_matches.setdefault(product2, set()).add(product1)

# Save
with open('text_matches.pkl', 'wb') as f:
    pickle.dump(text_matches, f)
print("Text matches saved!")    