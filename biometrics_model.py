# biometrics_model.py
# Prototype feature-extraction + demo classifier for emotion prediction.
# IMPORTANT: This is a prototyping/demo model using synthetic data.
# Replace with real labeled physiological datasets for production use.

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

EMOTIONS = [
    "anxious", "tense", "happy", "excited", "calm",
    "sad", "angry", "bored", "surprised", "relaxed", "focused"
]

def extract_features_from_window(window):
    """
    window: dict with optional keys: hr (list), eda (list), temp (list),
            acc (list of [x,y,z]), rr (list)
    returns: numpy array feature vector
    """
    feat = []
    # HR features
    hr = np.array(window.get("hr", [np.nan]), dtype=float)
    feat.append(np.nanmean(hr))
    feat.append(np.nanstd(hr))
    # HRV proxy (mean absolute diff)
    if hr.size > 1:
        diffs = np.diff(hr)
        feat.append(np.nanmean(np.abs(diffs)))
        feat.append(np.nanstd(diffs))
    else:
        feat += [0.0, 0.0]

    # EDA
    eda = np.array(window.get("eda", [np.nan]), dtype=float)
    feat.append(np.nanmean(eda))
    feat.append(np.nanstd(eda))
    feat.append((np.nanmax(eda) - np.nanmin(eda)) if eda.size>0 else 0.0)

    # Skin temp
    temp = np.array(window.get("temp", [np.nan]), dtype=float)
    feat.append(np.nanmean(temp))
    feat.append(np.nanstd(temp))

    # Respiration rate
    rr = np.array(window.get("rr", [np.nan]), dtype=float)
    feat.append(np.nanmean(rr))
    feat.append(np.nanstd(rr))

    # Accelerometer magnitude stats
    acc = np.array(window.get("acc", []), dtype=float)
    if acc.ndim == 2 and acc.shape[0] > 0:
        mags = np.linalg.norm(acc, axis=1)
        feat.append(np.nanmean(mags))
        feat.append(np.nanstd(mags))
        feat.append(np.nanmax(mags))
    else:
        feat += [0.0, 0.0, 0.0]

    # Replace NaNs/infs with 0
    feat = [0.0 if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else float(x) for x in feat]
    return np.array(feat, dtype=float)

def train_demo_model(save_path="bio_model.joblib"):
    """
    Train a simple RandomForest on synthetic data for prototyping.
    Writes a joblib file containing {'model': clf, 'emotions': EMOTIONS}.
    """
    rng = np.random.RandomState(42)
    X = []
    y = []
    n_per_class = 300
    n_features = len(extract_features_from_window({}))
    for i, emo in enumerate(EMOTIONS):
        # create synthetic clusters: shift means by emotion index
        for _ in range(n_per_class):
            base = rng.normal(loc=(i - len(EMOTIONS)/2), scale=1.0, size=n_features)
            noise = rng.normal(scale=0.5, size=n_features)
            sample = base + noise
            X.append(sample)
            y.append(i)
    X = np.vstack(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Demo model trained. Train acc:", clf.score(X_train, y_train), "Test acc:", clf.score(X_test, y_test))
    joblib.dump({"model": clf, "emotions": EMOTIONS}, save_path)
    return save_path

def load_model(path="bio_model.joblib"):
    return joblib.load(path)

def predict_emotion_from_window(window, model_dict):
    """
    window: same structure as extract_features_from_window input.
    model_dict: object returned by load_model (dict with 'model' and 'emotions')
    returns: dict { emotion, confidence, all: {emotion:score} }
    """
    X = extract_features_from_window(window).reshape(1, -1)
    clf = model_dict["model"]
    emotions = model_dict["emotions"]
    probs = clf.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    return {
        "emotion": emotions[idx],
        "confidence": float(probs[idx]),
        "all": {emotions[i]: float(probs[i]) for i in range(len(emotions))}
    }

if __name__ == "__main__":
    # If executed directly, train demo model
    train_demo_model("bio_model.joblib")

