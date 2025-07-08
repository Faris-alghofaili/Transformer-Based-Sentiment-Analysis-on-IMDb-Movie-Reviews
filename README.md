Objective:
This project aims to develop a sentiment analysis model using a Transformer-based
architecture (specifically BERT) to classify IMDb movie reviews as either positive or
negative. By leveraging a pre-trained model, we aim to save training time, improve
performance, and reduce the resources needed compared to building a model from
scratch.

Dataset Description:
• Source: IMDb 50K Reviews Dataset on Kaggle https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
• Size: 50,000 labeled reviews
• Classes: positive (1), negative (0)

Tools & Libraries:
• transformers — for using the BERT model and tokenizer
• datasets — for dataset handling (optional alternative)
• torch — PyTorch framework
• scikit-learn — metrics and preprocessing
• wordcloud, seaborn, matplotlib — for visualization
