import importlib

attr = importlib.import_module("attraction-classifier.infer").AttractionClassifier()
userEmbedder = importlib.import_module("simtoon-embeddings.user_embedder").UserEmbedder()
userUtils = importlib.import_module("simtoon-embeddings.user_utils").UserUtils()