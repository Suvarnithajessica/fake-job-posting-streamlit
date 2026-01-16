import pickle
v = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
print(hasattr(v, "idf_"))
