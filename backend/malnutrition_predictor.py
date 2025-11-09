import os
import cv2
import numpy as np
import mediapipe as mp
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)


class MalnutritionPredictor:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.model = None
        self.pca = None

        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )

    # ---------------- FACE FEATURES ----------------
    def extract_face_features(self, image):
        """Extract facial landmarks with absolute scaling"""
        h, w, _ = image.shape
        scale = max(h, w)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return [0] * 13

        landmarks = np.array([[lm.x * w, lm.y * h, lm.z * scale]
                              for lm in results.multi_face_landmarks[0].landmark])

        # Calculate distances in pixel space
        face_width = np.linalg.norm(landmarks[234] - landmarks[454])
        face_length = np.linalg.norm(landmarks[10] - landmarks[152])
        left_cheek_depth = np.mean([landmarks[i][2] for i in [123, 116, 117]])
        right_cheek_depth = np.mean([landmarks[i][2] for i in [352, 345, 346]])
        left_eye_depth = np.mean([landmarks[i][2] for i in [159, 145, 133]])
        right_eye_depth = np.mean([landmarks[i][2] for i in [386, 374, 362]])
        jaw_width_top = np.linalg.norm(landmarks[127] - landmarks[356])
        jaw_width_bottom = np.linalg.norm(landmarks[172] - landmarks[397])
        left_temple, right_temple = landmarks[21][2], landmarks[251][2]
        face_ratio = face_width / (face_length + 1e-6)

        landmarks_flat = landmarks.flatten()
        mean_coord, std_coord = np.mean(landmarks_flat), np.std(landmarks_flat)

        return [
            face_width, face_length, face_ratio,
            left_cheek_depth, right_cheek_depth,
            left_eye_depth, right_eye_depth,
            jaw_width_top, jaw_width_bottom,
            left_temple, right_temple,
            mean_coord, std_coord
        ]

    # ---------------- BODY FEATURES ----------------
    def extract_body_features(self, image):
        """Extract body pose landmarks with absolute scaling"""
        h, w, _ = image.shape
        scale = max(h, w)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        if not results.pose_landmarks:
            return [0] * 21

        landmarks = np.array([[lm.x * w, lm.y * h, lm.z * scale, lm.visibility]
                              for lm in results.pose_landmarks.landmark])

        # Distances in pixel space
        shoulder_width = np.linalg.norm(landmarks[11][:2] - landmarks[12][:2])
        torso_length = np.linalg.norm(landmarks[11][:2] - landmarks[23][:2])
        hip_width = np.linalg.norm(landmarks[23][:2] - landmarks[24][:2])

        left_arm_upper = np.linalg.norm(landmarks[11][:2] - landmarks[13][:2])
        left_arm_lower = np.linalg.norm(landmarks[13][:2] - landmarks[15][:2])
        right_arm_upper = np.linalg.norm(landmarks[12][:2] - landmarks[14][:2])
        right_arm_lower = np.linalg.norm(landmarks[14][:2] - landmarks[16][:2])

        left_leg_upper = np.linalg.norm(landmarks[23][:2] - landmarks[25][:2])
        left_leg_lower = np.linalg.norm(landmarks[25][:2] - landmarks[27][:2])
        right_leg_upper = np.linalg.norm(landmarks[24][:2] - landmarks[26][:2])
        right_leg_lower = np.linalg.norm(landmarks[26][:2] - landmarks[28][:2])

        shoulder_hip_ratio = shoulder_width / (hip_width + 1e-6)
        torso_shoulder_ratio = torso_length / (shoulder_width + 1e-6)

        left_arm_ratio = left_arm_upper / (left_arm_lower + 1e-6)
        right_arm_ratio = right_arm_upper / (right_arm_lower + 1e-6)
        left_leg_ratio = left_leg_upper / (left_leg_lower + 1e-6)
        right_leg_ratio = right_leg_upper / (right_leg_lower + 1e-6)

        avg_visibility = np.mean(landmarks[:, 3])
        limb_visibility = np.mean([landmarks[i][3]
                                   for i in [13, 14, 15, 16, 25, 26, 27, 28]])

        coords_flat = landmarks[:, :3].flatten()
        mean_coord, std_coord = np.mean(coords_flat), np.std(coords_flat)

        return [
            shoulder_width, torso_length, hip_width,
            left_arm_upper, left_arm_lower, right_arm_upper, right_arm_lower,
            left_leg_upper, left_leg_lower, right_leg_upper, right_leg_lower,
            shoulder_hip_ratio, torso_shoulder_ratio,
            left_arm_ratio, right_arm_ratio, left_leg_ratio, right_leg_ratio,
            avg_visibility, limb_visibility,
            mean_coord, std_coord
        ]

    # ---------------- FEATURE EXTRACTION ----------------
    def extract_features_from_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read image: {image_path}")
            return None

        face_features = self.extract_face_features(image)
        body_features = self.extract_body_features(image)
        features = face_features + body_features

        if all(f == 0 for f in features):
            print(f"⚠️ No landmarks detected in {os.path.basename(image_path)}")
            return None

        return features

    # ---------------- LOAD DATASET ----------------
    def load_dataset(self):
        data, labels = [], []
        for split in ['train', 'valid', 'test']:
            for label_name in ['healthy', 'malnurished']:
                folder_path = os.path.join(self.dataset_path, split, label_name)
                label = 0 if label_name == 'healthy' else 1
                if not os.path.exists(folder_path):
                    print(f"Warning: {folder_path} not found")
                    continue
                for img_file in os.listdir(folder_path):
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    path = os.path.join(folder_path, img_file)
                    features = self.extract_features_from_image(path)
                    if features is not None:
                        data.append(features)
                        labels.append(label)
                        print(f"Processed: {split}/{label_name}/{img_file}")
        return np.array(data), np.array(labels)

    # ---------------- FEATURE NAMES ----------------
    def create_feature_names(self):
        return [
            'face_width', 'face_length', 'face_ratio', 'left_cheek_depth', 'right_cheek_depth',
            'left_eye_depth', 'right_eye_depth', 'jaw_width_top', 'jaw_width_bottom',
            'left_temple', 'right_temple', 'face_mean_coord', 'face_std_coord',
            'shoulder_width', 'torso_length', 'hip_width',
            'left_arm_upper', 'left_arm_lower', 'right_arm_upper', 'right_arm_lower',
            'left_leg_upper', 'left_leg_lower', 'right_leg_upper', 'right_leg_lower',
            'shoulder_hip_ratio', 'torso_shoulder_ratio',
            'left_arm_ratio', 'right_arm_ratio', 'left_leg_ratio', 'right_leg_ratio',
            'avg_visibility', 'limb_visibility', 'body_mean_coord', 'body_std_coord'
        ]

    # ---------------- PREPROCESSING ----------------
    def preprocess_data(self, X_train, X_test, X_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)

        print("\nApplying PCA for dimensionality reduction...")
        self.pca = PCA(n_components=0.95)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        print(f"Reduced features: {X_train_pca.shape[1]} kept (95% variance)\n")

        return X_train_pca, X_test_pca, X_val_pca

    # ---------------- TRAIN MODEL ----------------
    def train_model(self, X_train, y_train, X_val, y_val):
        """Automatic handling for early stopping across XGBoost versions"""
        version = tuple(map(int, xgb.__version__.split('.')[:2]))
        print(f"Using XGBoost v{'.'.join(map(str, version))}")

        params = dict(
            n_estimators=1200,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1.0,
            reg_lambda=2.0,
            reg_alpha=1.0,
            min_child_weight=5,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False
        )


        if version >= (1, 6) and version < (2, 0):
            print("→ Using sklearn API with early stopping")
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
        else:
            print("→ Using native XGBoost API for compatibility")
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=100
            )

        print("\nModel training complete.")
        return self.model

    # ---------------- EVALUATION ----------------
    def evaluate_model(self, X_test, y_test, feature_names):
        print("\nEvaluating model...")
        if isinstance(self.model, xgb.Booster):
            preds = self.model.predict(xgb.DMatrix(X_test))
            y_pred = (preds > 0.5).astype(int)
            y_pred_proba = preds
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('ROC Curve')
        plt.show()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    # ---------------- SAVE / LOAD ----------------
    def save_model(self, model_path='malnutrition_model.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'pca': self.pca}, f)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path='malnutrition_model.pkl'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.pca = data['pca']
        print(f"Model loaded from {model_path}")

    # ---------------- PREDICTION ----------------
    def predict_new_image(self, image_path):
        """Predict malnutrition status for a new image using trained model"""
        if self.model is None or self.scaler is None:
            print("❌ Model or scaler not loaded. Please call load_model() first.")
            return None, None

        # Extract features
        features = self.extract_features_from_image(image_path)
        if features is None or all(f == 0 for f in features):
            print("⚠️ Warning: Could not detect face or body landmarks properly.")
            return None, None

        # Preprocess (scale + PCA)
        features_scaled = self.scaler.transform([features])
        if self.pca:
            features_scaled = self.pca.transform(features_scaled)

        # Predict
        if isinstance(self.model, xgb.Booster):
            preds = self.model.predict(xgb.DMatrix(features_scaled))
            prob = float(preds[0])
            label = 1 if prob > 0.5 else 0
        else:
            prob = float(self.model.predict_proba(features_scaled)[0, 1])
            label = int(self.model.predict(features_scaled)[0])

        # Interpret result
        result = "Malnourished" if label == 1 else "Healthy"
        confidence = prob * 100 if label == 1 else (1 - prob) * 100

        print(f"\n🧾 Prediction Result for {os.path.basename(image_path)}:")
        print(f"   → Class: {result}")
        print(f"   → Confidence: {confidence:.2f}%")

        return result, confidence
