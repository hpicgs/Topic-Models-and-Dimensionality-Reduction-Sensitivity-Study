def download_nltk_resources():
    import nltk
    import os

    nltk.download('words', download_dir=os.getcwd())
    nltk.download('stopwords', download_dir=os.getcwd())
    nltk.download('reuters', download_dir=os.getcwd())
    nltk.data.path.append(os.getcwd())
