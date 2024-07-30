import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import threading
import time

app = Flask(__name__)

# Initialize feature extractor and paths
fe = FeatureExtractor()
features = []
img_paths = []
feature_dir = Path("./static/feature")
img_dir = Path("./static/img")
uploaded_dir = Path("./static/uploaded")


def reload_features():
    global features, img_paths
    features = []
    img_paths = []

    # Clean up uploaded directory
    clean_uploaded_dir()

    # Load features and corresponding image paths
    for feature_path in feature_dir.glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(img_dir / (feature_path.stem + ".jpg"))
    features = np.array(features)
    print(f"Reloaded features at {datetime.now()}")


def clean_uploaded_dir():
    # Get all files in uploaded directory
    uploaded_files = list(uploaded_dir.glob("*"))

    # Sort files by creation time (oldest first)
    uploaded_files.sort(key=os.path.getctime)

    # Delete all but the most recent three files
    files_to_delete = uploaded_files[:-3]
    for file in files_to_delete:
        file.unlink()


# Initial load of features
reload_features()


# Background thread to reload features every 15 seconds
def reload_thread():
    while True:
        time.sleep(15)
        reload_features()


reload_thread = threading.Thread(target=reload_thread)
reload_thread.daemon = True
reload_thread.start()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = str(uploaded_dir / (datetime.now().isoformat().replace(":", ".") + "_" + file.filename))
        img.save(uploaded_img_path)

        # Run search with updated features
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0")
