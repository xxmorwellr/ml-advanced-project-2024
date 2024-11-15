import pandas as pd
# pip install git+https://github.com/openai/CLIP.git
# pip install torch torchvision
import torch
import clip
from PIL import Image

## 0. read train/test text file
train_df = pd.read_csv("train_subset.csv")
# split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
test_df = pd.read_csv("test.csv") # only 3 items
tmp = train_df.groupby('label_group').posting_id.agg('unique').to_dict()
# print(tmp)
# tmp = {
#     label_group_1: ['posting_id_1', 'posting_id_2'],
#     label_group_2: ['posting_id_3', 'posting_id_4', 'posting_id_5']
# }

# look into...
# print(train_df.head())
## provided fields: posting_id image image_phash title  label_group

## 1. utilize CLIP model
# 1.1 load model and pre-trained wordlsit
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 1.2 preprocess text data

# convert encoding format(utf-8 => unicode, "\xc3\x89" represents "É")
# 
# byte_string = r'\xf0\x9f\x87\xb2\xf0\x9f\x87\xa8MG\xf0\x9f\x87\xb2\xf0\x9f\x87\xa8 Jam Tangan  Fashion Wanita Rantai Strap Emas Silver Dial Logam Fashion JT12'
# byte_string = r'\xe2\x9d\xa4 Ewalook\xe2\x9d\xa4'
# byte_string = b'Nescafe \xc3\x89clair Latte 220ml'
# decoded_byte = bytes(byte_string, 'latin1').decode('unicode_escape')
# decoded_string = decoded_byte.encode('latin1').decode('utf-8')
# print(decoded_string)  # output: Nescafe Éclair Latte 220ml

def clean_title(title):
    if isinstance(title, str):
        # Process by replacing and handling escape characters
        # First, convert escape sequences in the string to bytes
        bytes_title = bytes(title, 'latin1').decode('unicode_escape')
        # Then, decode bytes into a UTF-8 string
        return bytes_title.encode('latin1').decode('utf-8')
    return title


# Apply to DataFrame
train_df['title'] = train_df['title'].apply(clean_title)
descriptions = train_df['title'].tolist()
# lengths = [len(token) for token in descriptions]
# print(f"Max token length: {max(lengths)}")  # 打印最大 token 长度:252 (phone case + several models)
# Sort descriptions by length in descending order and get the top 10
# top_10_descriptions = sorted(descriptions, key=len, reverse=True)[:10]
# # Print the top 10 descriptions
# for desc in top_10_descriptions:
#     print(desc)

# CLIP Model has character lengh limit..
# Convert the 'title' column to a list and truncate to the max length for CLIP
max_length = 77  # Adjust this according to the model's limit
descriptions = [desc[:max_length] for desc in train_df['title'].tolist()]

# print(descriptions)  # Print the processed descriptions

texts = clip.tokenize(descriptions).to(device)


# 1.3 process image data
image_paths = train_df['image'].apply(lambda x: f"train_images/{x}")
images = [preprocess(Image.open(img)).unsqueeze(0).to(device) for img in image_paths]

# calculate embeddings
with torch.no_grad():
    image_features = torch.cat([model.encode_image(img) for img in images])
    text_features = torch.cat([model.encode_image(text) for text in texts])

# normalize
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
