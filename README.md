# Shopee Product Matching
This repository contains my final project for the master's course *Machine Learning for Statistical NLP: Advanced*. The project is open-ended, and I selected a previous Kaggle competition as the focus of my exploration.

## Basic File Structure
- `text.py`  
Extracts text features using TF-IDF and BERT models and identifies text matches based on cosine similarity.

- `image.py`  
Extracts image features using EfficientNet with ArcFace and finds image matches using the KNN method.

- `combine.py`  
Combines text and image matches and evaluates the results against the ground truth label mappings.

- `clip.py` **(potential)**  
Implements feature extraction using the CLIP model.

- `train_subset.csv`    
The random 10% subset data I used in this implemention.

Due to file size limitation, I don't upload metadata here. You can download metadata using Kaggle api: `kaggle competitions download -c shopee-product-matching`.

## Limitations of Process Control 
To implement cross-validation, the ideal structure for managing the complex processes would involve a main script coordinating three or more sub-scripts. However, challenges in argument sharing and dynamic parameter adjustments based on varying data sizes prevented full implementation of this structure at this time.

## Source
Kaggle competition: [Shopee Product Matching](https://www.kaggle.com/competitions/shopee-product-matching).  
