# ============================================================
# Automated Cardio-Respiratory Sound Analysis for Disease Screening
# Author: Arige Bharath Kumar
# UI: Streamlit (Modern Web App)
# Run: streamlit run main.py
# ============================================================

import os
import joblib
import numpy as np
import soundfile as sf
import resampy
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn_lvq import GlvqModel
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Cardio-Respiratory Sound Analysis",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1rem; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "X"          not in st.session_state: st.session_state.X          = None
if "Y"          not in st.session_state: st.session_state.Y          = None
if "x_train"    not in st.session_state: st.session_state.x_train    = None
if "x_test"     not in st.session_state: st.session_state.x_test     = None
if "y_train"    not in st.session_state: st.session_state.y_train    = None
if "y_test"     not in st.session_state: st.session_state.y_test     = None
if "categories" not in st.session_state: st.session_state.categories = None
if "file_name"  not in st.session_state: st.session_state.file_name  = None
if "metrics"    not in st.session_state:
    st.session_state.metrics = {
        "names": [], "accuracy": [], "precision": [], "recall": [], "fscore": []
    }

model_folder = "model"
os.makedirs(model_folder, exist_ok=True)


# ============================================================
# AUDIO HELPER FUNCTIONS
# ============================================================
def load_and_preprocess_audio(file_path, target_sr=16000):
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    return audio.astype(np.float32), target_sr


def extract_audio_features(audio, sr):
    mfccs  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=35)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1)])


# ============================================================
# METRICS FUNCTION
# ============================================================
def Calculate_Metrics(algorithm, predict, y_test, predict_proba=None):
    categories = st.session_state.categories

    a = accuracy_score(y_test, predict)                    * 100
    p = precision_score(y_test, predict, average='macro')  * 100
    r = recall_score(y_test, predict,    average='macro')  * 100
    f = f1_score(y_test, predict,        average='macro')  * 100

    st.session_state.metrics["names"].append(algorithm)
    st.session_state.metrics["accuracy"].append(a)
    st.session_state.metrics["precision"].append(p)
    st.session_state.metrics["recall"].append(r)
    st.session_state.metrics["fscore"].append(f)

    st.markdown(f"### 📊 {algorithm} — Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Accuracy",  f"{a:.2f}%")
    c2.metric("🎯 Precision", f"{p:.2f}%")
    c3.metric("🔁 Recall",    f"{r:.2f}%")
    c4.metric("⚖️ F1-Score",  f"{f:.2f}%")

    CR = classification_report(y_test, predict, target_names=categories)
    with st.expander("📋 Classification Report"):
        st.text(CR)

    conf_matrix = confusion_matrix(y_test, predict)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories,
                annot=True, cmap="magma", fmt="g", ax=ax)
    ax.set_title(f"{algorithm} — Confusion Matrix")
    ax.set_ylabel("True Class")
    ax.set_xlabel("Predicted Class")
    st.pyplot(fig)
    plt.close()

    if predict_proba is not None:
        y_test_flat = np.array(y_test).flatten()
        y_test_bin  = label_binarize(y_test_flat, classes=np.arange(len(categories)))
        if np.unique(y_test_flat).shape[0] < 2:
            st.warning("ROC AUC: Not defined — only one class in test set.")
            return
        try:
            roc_auc = roc_auc_score(y_test_bin, predict_proba,
                                    average='macro', multi_class='ovr') * 100
            st.info(f"📈 ROC AUC Score: **{roc_auc:.2f}%**")

            fig2, ax2 = plt.subplots(figsize=(7, 5))
            for i in range(len(categories)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], predict_proba[:, i])
                roc_auc_i   = auc(fpr, tpr)
                ax2.plot(fpr, tpr, label=f'{categories[i]} (AUC={roc_auc_i:.2f})')
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title(f"{algorithm} — ROC Curves")
            ax2.legend(loc='lower right')
            ax2.grid(True)
            st.pyplot(fig2)
            plt.close()

        except ValueError as e:
            st.error(f"ROC AUC could not be computed: {str(e)}")


# ============================================================
# HEADER
# ============================================================
st.markdown("""
    <h1 style='text-align:center; color:#1a3c5e;'>
        🫀 Automated Cardio-Respiratory Sound Analysis
    </h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Disease Screening using Deep Learning & Audio Feature Extraction
    </p>
    <hr>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# STEP 1
st.sidebar.subheader("📁 Step 1: Upload Dataset")
dataset_path = st.sidebar.text_input(
    "Paste dataset folder path:",
    placeholder="e.g. C:/dataset"
)
if dataset_path and os.path.isdir(dataset_path):
    categories = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]
    if categories:
        st.session_state.file_name  = dataset_path
        st.session_state.categories = categories
        np.save(os.path.join(model_folder, "categories.npy"), np.array(categories))
        st.sidebar.success(f"✅ {len(categories)} classes loaded!")
        st.sidebar.write(categories)
    else:
        st.sidebar.error("No class subfolders found.")
elif dataset_path:
    st.sidebar.error("Invalid path. Please check and try again.")
st.sidebar.markdown("---")
# STEP 2
st.sidebar.subheader("🎵 Step 2: Feature Extraction")
if st.sidebar.button("⚙️ Extract MFCC & Chroma Features"):
    if st.session_state.file_name is None:
        st.sidebar.error("Please load dataset first.")
    else:
        with st.spinner("Extracting features from audio files..."):
            if os.path.exists("X.npy") and os.path.exists("Y.npy"):
                st.session_state.X = np.load("X.npy")
                st.session_state.Y = np.load("Y.npy")
                st.sidebar.success("✅ Features loaded from cache!")
            else:
                features     = []
                labels       = []
                categories   = st.session_state.categories
                class_to_idx = {name: idx for idx, name in enumerate(categories)}

                for class_name in categories:
                    class_dir = os.path.join(st.session_state.file_name, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    for file in os.listdir(class_dir):
                        if file.lower().endswith('.wav'):
                            file_path = os.path.join(class_dir, file)
                            audio, sr = load_and_preprocess_audio(file_path)
                            if audio.size == 0:
                                continue
                            features.append(extract_audio_features(audio, sr))
                            labels.append(class_to_idx[class_name])

                st.session_state.X = np.array(features)
                st.session_state.Y = np.array(labels)
                np.save("X.npy", st.session_state.X)
                np.save("Y.npy", st.session_state.Y)
                st.sidebar.success("✅ Feature extraction complete!")

        st.sidebar.info(f"Total samples: {len(st.session_state.X)}")

st.sidebar.markdown("---")

# STEP 3
st.sidebar.subheader("✂️ Step 3: Train/Test Split")
if st.sidebar.button("🔀 Split Dataset (80/20)"):
    if st.session_state.X is None:
        st.sidebar.error("Please extract features first.")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            st.session_state.X, st.session_state.Y, test_size=0.20, random_state=42
        )
        st.session_state.x_train = x_train
        st.session_state.x_test  = x_test
        st.session_state.y_train = y_train
        st.session_state.y_test  = y_test
        st.sidebar.success("✅ Split complete!")
        st.sidebar.write(f"Train: {x_train.shape[0]} | Test: {x_test.shape[0]}")

st.sidebar.markdown("---")

# STEP 4
st.sidebar.subheader("🤖 Step 4: Train Models")
run_glvq       = st.sidebar.button("🔷 Train GLVQ")
run_perceptron = st.sidebar.button("🔶 Train Perceptron")
run_dnn        = st.sidebar.button("🧠 Train DNN")
run_ffbp       = st.sidebar.button("⚡ Train FFBP-SVM")

st.sidebar.markdown("---")

# STEP 5
st.sidebar.subheader("📊 Step 5: Compare Models")
run_compare = st.sidebar.button("📊 Compare All Models")
st.sidebar.markdown("---")

# STEP 6
st.sidebar.subheader("🔮 Step 6: Predict")
uploaded_wav = st.sidebar.file_uploader("Upload a .wav file", type=["wav"])
run_predict  = st.sidebar.button("🎯 Run Prediction")


# ============================================================
# MAIN AREA
# ============================================================
if st.session_state.categories:
    st.success(f"✅ Dataset loaded — Classes: {st.session_state.categories}")
if st.session_state.x_train is not None:
    st.info(f"✅ Data split — Train: {len(st.session_state.x_train)} | Test: {len(st.session_state.x_test)}")

st.markdown("---")

# ── GLVQ ───────────────────────────────────────────────────
if run_glvq:
    if st.session_state.x_train is None:
        st.error("Please complete Train/Test Split first.")
    else:
        with st.spinner("Training GLVQ model..."):
            model_filename = os.path.join(model_folder, "Existing_GLVQ_model.pkl")
            if os.path.exists(model_filename):
                mlmodel = joblib.load(model_filename)
                st.info("Loaded existing GLVQ model.")
            else:
                mlmodel = GlvqModel()
                mlmodel.fit(st.session_state.x_train, st.session_state.y_train)
                joblib.dump(mlmodel, model_filename)

            y_pred = mlmodel.predict(st.session_state.x_test)
            try:
                y_proba = mlmodel.predict_proba(st.session_state.x_test)
            except AttributeError:
                y_proba = None

        Calculate_Metrics("GLVQ", y_pred, st.session_state.y_test, y_proba)

# ── Perceptron ─────────────────────────────────────────────
if run_perceptron:
    if st.session_state.x_train is None:
        st.error("Please complete Train/Test Split first.")
    else:
        with st.spinner("Training Perceptron model..."):
            model_filename = os.path.join(model_folder, "Existing_Perceptron_model.pkl")
            if os.path.exists(model_filename):
                mlmodel = joblib.load(model_filename)
                st.info("Loaded existing Perceptron model.")
            else:
                mlmodel = Perceptron(max_iter=8, eta0=1.0, random_state=42)
                mlmodel.fit(st.session_state.x_train, st.session_state.y_train)
                joblib.dump(mlmodel, model_filename)

            y_pred = mlmodel.predict(st.session_state.x_test)

        Calculate_Metrics("Perceptron", y_pred, st.session_state.y_test, None)

# ── DNN ────────────────────────────────────────────────────
if run_dnn:
    if st.session_state.x_train is None:
        st.error("Please complete Train/Test Split first.")
    else:
        with st.spinner("Training DNN model... (this may take a minute)"):
            x_train     = st.session_state.x_train
            x_test      = st.session_state.x_test
            y_train     = st.session_state.y_train
            y_test      = st.session_state.y_test
            y_train_cat = to_categorical(y_train)
            y_test_cat  = to_categorical(y_test)
            input_dim   = x_train.shape[1]
            output_dim  = y_train_cat.shape[1]
            model_path  = os.path.join(model_folder, "DNN_model.keras")

            dnn_model = None
            if os.path.exists(model_path):
                try:
                    dnn_model = load_model(model_path)
                    st.info("Loaded existing DNN model.")
                except Exception:
                    dnn_model = None

            if dnn_model is None:
                dnn_model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(128, activation='relu'),
                    Dense(64,  activation='relu'),
                    Dense(32,  activation='relu'),
                    Dense(output_dim, activation='softmax')
                ])
                dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                dnn_model.fit(x_train, y_train_cat,
                              validation_data=(x_test, y_test_cat),
                              epochs=50, batch_size=32, verbose=0,
                              callbacks=[early_stop])
                dnn_model.save(model_path)
                st.info("DNN model trained and saved.")

            y_pred        = dnn_model.predict(x_test)
            y_pred_labels = y_pred.argmax(axis=1)

        Calculate_Metrics("DNN Classifier", y_pred_labels, st.session_state.y_test, y_pred)

# ── FFBP-SVM ───────────────────────────────────────────────
if run_ffbp:
    if st.session_state.x_train is None:
        st.error("Please complete Train/Test Split first.")
    else:
        with st.spinner("Training FFBP-SVM model... (this may take a few minutes)"):
            x_train     = st.session_state.x_train
            x_test      = st.session_state.x_test
            y_train     = st.session_state.y_train
            y_test      = st.session_state.y_test
            y_train_cat = to_categorical(y_train)
            y_test_cat  = to_categorical(y_test)
            input_dim   = x_train.shape[1]

            extractor_path = os.path.join(model_folder, "DNN_feature_extractor.keras")
            svm_path       = os.path.join(model_folder, "DNN_SVM_model.pkl")

            feature_extractor = None
            if os.path.exists(extractor_path):
                try:
                    feature_extractor = load_model(extractor_path)
                    st.info("Loaded existing DNN Feature Extractor.")
                except Exception as e:
                    st.warning(f"Could not load extractor: {str(e)} — Retraining...")
                    feature_extractor = None

            if feature_extractor is None:
                st.info("Training DNN Feature Extractor...")
                full_model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(128, activation='relu'),
                    Dense(64,  activation='relu'),
                    Dense(32,  activation='relu'),
                    Dense(y_train_cat.shape[1], activation='softmax')
                ])
                full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # FIX 1: increased patience and epochs, reduced batch size
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                full_model.fit(x_train, y_train_cat,
                               validation_data=(x_test, y_test_cat),
                               epochs=150, batch_size=16, verbose=0,
                               callbacks=[early_stop])

                feature_extractor = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(128, activation='relu'),
                    Dense(64,  activation='relu'),
                    Dense(32,  activation='relu')
                ])
                for i, layer in enumerate(feature_extractor.layers):
                    layer.set_weights(full_model.layers[i].get_weights())

                feature_extractor.save(extractor_path)
                st.info("Feature extractor trained and saved.")

            x_train_feat = feature_extractor.predict(x_train)
            x_test_feat  = feature_extractor.predict(x_test)

            if os.path.exists(svm_path):
                svm_model = joblib.load(svm_path)
                st.info("Loaded existing SVM model.")
            else:
                st.info("Training SVM classifier...")
                svm_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svc", SVC(
                        kernel='rbf',
                        C=10,           # FIX 2: changed from 50 to 10
                        gamma='scale',  # FIX 2: changed from 0.01 to scale
                        probability=True,
                        class_weight="balanced",
                        cache_size=1000
                    ))
                ])
                svm_model.fit(x_train_feat, y_train)
                joblib.dump(svm_model, svm_path)
                st.info("SVM model trained and saved.")

            y_pred  = svm_model.predict(x_test_feat)
            y_proba = svm_model.predict_proba(x_test_feat) if hasattr(svm_model, "predict_proba") else None

        Calculate_Metrics("FFBP-SVM", y_pred, st.session_state.y_test, y_proba)
# ── Compare Models ─────────────────────────────────────────
if run_compare:
    names = st.session_state.metrics["names"]
    if len(names) == 0:
        st.error("Please train at least one model first.")
    else:
        st.markdown("## 📊 Model Comparison")

        metrics_data = {
            "Accuracy":  st.session_state.metrics["accuracy"],
            "Precision": st.session_state.metrics["precision"],
            "Recall":    st.session_state.metrics["recall"],
            "F1-Score":  st.session_state.metrics["fscore"],
        }

        x      = np.arange(len(names))
        width  = 0.2
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric, color=colors[i])

        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### 📋 Summary Table")
        df = pd.DataFrame({
            "Model":     names,
            "Accuracy":  [f"{v:.2f}%" for v in st.session_state.metrics["accuracy"]],
            "Precision": [f"{v:.2f}%" for v in st.session_state.metrics["precision"]],
            "Recall":    [f"{v:.2f}%" for v in st.session_state.metrics["recall"]],
            "F1-Score":  [f"{v:.2f}%" for v in st.session_state.metrics["fscore"]],
        })
        st.dataframe(df, use_container_width=True)

        best_idx   = st.session_state.metrics["accuracy"].index(max(st.session_state.metrics["accuracy"]))
        best_model = names[best_idx]
        best_acc   = st.session_state.metrics["accuracy"][best_idx]
        st.success(f"🏆 Best Model: **{best_model}** with Accuracy: **{best_acc:.2f}%**")

# ── Prediction ─────────────────────────────────────────────
if run_predict:
    if uploaded_wav is None:
        st.error("Please upload a .wav file first using the sidebar.")
    else:
        categories_path = os.path.join(model_folder, "categories.npy")
        if os.path.exists(categories_path):
            st.session_state.categories = np.load(categories_path, allow_pickle=True).tolist()
        elif st.session_state.categories is None:
            st.session_state.categories = ['Aortic Stenosis', 'COPD']

        categories = st.session_state.categories
        temp_path  = os.path.join(model_folder, "temp_input.wav")

        with open(temp_path, "wb") as f:
            f.write(uploaded_wav.read())

        with st.spinner("Analyzing audio..."):
            audio, sr = load_and_preprocess_audio(temp_path)

            if audio.size == 0:
                st.error("Audio loading failed.")
            else:
                raw_features   = extract_audio_features(audio, sr).reshape(1, -1)
                extractor_path = os.path.join(model_folder, "DNN_feature_extractor.keras")
                svm_path       = os.path.join(model_folder, "DNN_SVM_model.pkl")

                if not os.path.exists(extractor_path) or not os.path.exists(svm_path):
                    st.error("Please train the FFBP-SVM model first before predicting.")
                else:
                    feature_extractor = load_model(extractor_path)
                    dnn_features      = feature_extractor.predict(raw_features)
                    svm_model         = joblib.load(svm_path)
                    prediction        = svm_model.predict(dnn_features)[0]
                    predicted_class   = categories[prediction] if prediction < len(categories) else str(prediction)

                    st.markdown("---")
                    st.markdown("## 🔮 Prediction Result")
                    st.success(f"### 🩺 Predicted Disease: **{predicted_class}**")
                    st.audio(temp_path, format="audio/wav")

                    y_audio, sr_audio = librosa.load(temp_path, sr=None)

                    fig, ax = plt.subplots(figsize=(10, 3))
                    librosa.display.waveshow(y_audio, sr=sr_audio, ax=ax, color="#1a3c5e")
                    ax.set_title(f"Waveform — Predicted: {predicted_class}", fontsize=13, fontweight='bold')
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    fig2, ax2 = plt.subplots(figsize=(10, 3))
                    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr_audio, n_mfcc=35)
                    img   = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
                    fig2.colorbar(img, ax=ax2)
                    ax2.set_title("MFCC Features", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:gray;'>
        Built by <b>Arige Bharath Kumar</b> |
        B.Tech CSE (Data Science) — Malla Reddy Engineering College
    </p>
""", unsafe_allow_html=True)