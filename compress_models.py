import pickle
import gzip

models = [
    "knn_model.pkl",
    "naive_bayes_model.pkl",
    "decision_tree_model.pkl",
    "tfidf_vectorizer.pkl"
]

for model in models:
    with open(model, "rb") as f:
        obj = pickle.load(f)

    with gzip.open(model + ".gz", "wb") as f:
        pickle.dump(obj, f)

print("All models compressed successfully.")
