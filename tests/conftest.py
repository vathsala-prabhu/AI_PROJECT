import os

# Tests use the fast keyword engine so they never load the transformer.
os.environ["MOODMENTOR_EMOTION_ENGINE"] = "keyword"