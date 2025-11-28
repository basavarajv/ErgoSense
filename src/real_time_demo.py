# real_time_inference.py (ErgoSense - Updated for New Dataset)
"""
Real-time posture inference for ErgoSense.

Requirements:
 - models/bilstm_model.pth          (trained PyTorch BiLSTM)
 - models/svm_pipeline.pkl          (trained sklearn Pipeline with scaler + SVC)
 - models/scaler.pkl                (saved normalization scaler)
 - dataset/posture_dataset.csv      (used during training for column order)
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from collections import deque
import time


# Settings

SEQ_LEN = 30          # frames per sequence
ALPHA = 0.6           # ensemble weight for BiLSTM probabilities
SMOOTH_WINDOW = 5     # rolling average for stability
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Upper-body landmarks used

UPPER_BODY_LANDMARKS = {
    0:  "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    27: "left_ear",
    28: "right_ear"
}
COL_ORDER = []
for _, name in UPPER_BODY_LANDMARKS.items():
    COL_ORDER.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])


# BiLSTM model definition

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        emb = out.mean(dim=1)
        logits = self.classifier(emb)
        return logits, emb

# --------------------------
# RL Agent (DQN)
# --------------------------
class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_rl_agent(path, state_dim, action_dim):
    agent = DQNNet(state_dim, action_dim)
    agent.load_state_dict(torch.load(path, map_location=DEVICE))
    agent.eval()
    return agent




# Helper functions

def load_models(model_dir=MODEL_DIR):
    bilstm_path = os.path.join(model_dir, "bilstm_model.pth")
    svm_path = os.path.join(model_dir, "svm_pipeline.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    rl_agent_path = "models/dqn_agent.pth"

    # Check model files
    for path in [bilstm_path, svm_path, scaler_path,rl_agent_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    input_size = len(COL_ORDER)  # 44 features
    model = BiLSTMEncoder(input_size=input_size)
    model.load_state_dict(torch.load(bilstm_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)

    # RL parameters
    ACTION_DIM = 4  # (0 = no action, 1 = soft, 2 = strong, 3 = specific)
    STATE_DIM = 128 + 3  # (embedding size + label + bad_history + confidence)

    rl_agent = load_rl_agent(rl_agent_path, STATE_DIM, ACTION_DIM)

    
    print(" Loaded BiLSTM, SVM, and Scaler successfully.")
    return model, svm, scaler,

def extract_upper_body_from_results(results):
    """Extract 44 upper-body landmark features (x,y,z,visibility)."""
    vals = []
    if not results.pose_landmarks:
        return [0.0] * len(COL_ORDER)
    for idx in UPPER_BODY_LANDMARKS.keys():
        lm = results.pose_landmarks.landmark[idx]
        vals.extend([lm.x, lm.y, lm.z, lm.visibility])
    return vals

def predict_from_sequence(model, svm_pipeline, scaler, seq_array, device=DEVICE, alpha=ALPHA):
    """Run BiLSTM + SVM ensemble prediction."""
    # Normalize sequence using same scaler
    seq_flat = seq_array.reshape(-1, seq_array.shape[-1])
    seq_scaled = scaler.transform(seq_flat)
    seq_array = seq_scaled.reshape(seq_array.shape)

    x = torch.tensor(seq_array, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, emb = model(x)
        probs_bilstm = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emb_np = emb.cpu().numpy()
        probs_svm = svm_pipeline.predict_proba(emb_np)[0]

        # Ensemble average
        probs = alpha * probs_bilstm + (1 - alpha) * probs_svm
        pred = int(np.argmax(probs))
    return pred, probs, probs_bilstm, probs_svm



# --------------------------
# RL State Builder
# --------------------------
from collections import deque
bad_history = deque(maxlen=30)

def build_rl_state(embedding, label, confidence):
    # update bad history
    bad_history.append(label)
    recent_bad_fraction = sum(bad_history) / len(bad_history)

    # embedding is already a numpy array → flatten it
    emb = embedding.flatten()

    # RL state = [embedding, label, bad_history_ratio, confidence]
    state = np.concatenate([
        emb,
        np.array([label, recent_bad_fraction, confidence], dtype=np.float32)
    ])
    return state.astype(np.float32)



# Real-time inference loop

def main():
    print(" Starting ErgoSense real-time posture detection...")
    try:
        model, svm, scaler = load_models()
    except FileNotFoundError as e:
        print(f" {e}")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    buffer = deque(maxlen=SEQ_LEN)
    pred_smooth = deque(maxlen=SMOOTH_WINDOW)
    label_map = {0: "GOOD POSTURE", 1: "BAD POSTURE"}

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Camera not detected.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract & buffer
        features = extract_upper_body_from_results(results)
        buffer.append(features)

        display_text = "Analyzing posture..."
        color = (200, 200, 0)

        if len(buffer) == SEQ_LEN:
            seq_np = np.array(buffer)
            pred, probs, probs_bi, probs_svm = predict_from_sequence(model, svm, scaler, seq_np)
            
            # Build RL state
            embedding = emb_np  # this already exists inside predict_from_sequence
            label = pred
            confidence = probs[pred]

            rl_state = build_rl_state(embedding, label, confidence)

            # Convert to tensor and get action
            state_tensor = torch.tensor(rl_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = rl_agent(state_tensor)
                action = int(q_vals.argmax().item())

            
            pred_smooth.append(pred)

            # --------------------------
            # Execute RL action
            # --------------------------

            if action == 1:
                cv2.putText(frame, "Soft Reminder: Align your back", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

            elif action == 2:
                cv2.putText(frame, "Strong Reminder!", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            elif action == 3:
                cv2.putText(frame, "Specific Advice: Straighten your neck", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0), 2)

            # Rolling average prediction
            final_pred = int(round(np.mean(pred_smooth)))
            conf = probs[final_pred]

            display_text = f"{label_map[final_pred]} ({conf*100:.1f}%)"
            color = (0, 255, 0) if final_pred == 0 else (0, 0, 255)

        # Overlay info
        cv2.rectangle(frame, (0, 0), (450, 70), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


        # --------------------------
        # Log RL experience
        # --------------------------
        import csv
        log_path = "logs/rl_memory.csv"
        os.makedirs("logs", exist_ok=True)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),                # timestamp
                label,                      # posture label
                confidence,                 # confidence
                action,                     # RL action
                np.mean(bad_history)        # recent bad fraction
            ])


        cv2.imshow("ErgoSense Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            buffer.clear()
            pred_smooth.clear()
            print(" Cleared frame buffer.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
