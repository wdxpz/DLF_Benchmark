import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/horse2zebra")
DEMO_DIR = os.path.join(BASE_DIR, "data/test_horse")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULT_DIR = os.path.join(BASE_DIR, "result")