# real_time_demo.py
"""
ErgoSense Real-Time Inference Demo (BiLSTM + SVM + RL + Split Screen UI)

"""

import os
import time
import csv
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ------------------------
# Global Settings
# ------------------------
SEQ_LEN = 30
ALPHA = 0.6              # Ensemble weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

PANEL_WIDTH = 350         # RL Panel width

# ------------------------
# Mediapose Landmarks
# ------------------------
UPPER_BODY_LMS = {
    0: "nose",
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
for _, name in UPPER_BODY_LMS.items():
    COL_ORDER += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]

FEATURE_DIM = len(COL_ORDER)  # 44

# ================================================================
# BiLSTM (must match training model: uses self.lstm)
# ================================================================
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        emb = out.mean(dim=1)
        logits = self.classifier(emb)
        return logits, emb

# ================================================================
# RL Agent (DQN)
# ================================================================
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

# ================================================================
# Load Models
# ================================================================
def load_models():
    bilstm_path = os.path.join(MODEL_DIR, "bilstm_model.pth")
    svm_path = os.path.join(MODEL_DIR, "svm_pipeline.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    dqn_path = os.path.join(MODEL_DIR, "dqn_agent.pth")

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load SVM
    svm = joblib.load(svm_path)

    # Load BiLSTM
    bilstm = BiLSTMEncoder(FEATURE_DIM)
    bilstm.load_state_dict(torch.load(bilstm_path, map_location=DEVICE))
    bilstm.to(DEVICE).eval()

    # Determine embedding size
    dummy = torch.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=torch.float32, device=DEVICE)
    _, emb = bilstm(dummy)
    emb_dim = emb.shape[1]

    # Load RL agent if available
    rl_agent = None
    if os.path.exists(dqn_path):
        state_dim = emb_dim + 3
        action_dim = 4
        rl_agent = DQNNet(state_dim, action_dim).to(DEVICE)
        rl_agent.load_state_dict(torch.load(dqn_path, map_location=DEVICE))
        rl_agent.eval()
        print("RL agent loaded.")
    else:
        print("No RL agent found → RL disabled.")

    print("Models loaded successfully.")
    return bilstm, svm, scaler, rl_agent, emb_dim

# ================================================================
# Feature Extraction
# ================================================================
def extract_upper_body(results):
    if not results.pose_landmarks:
        return [0.0] * FEATURE_DIM

    vals = []
    for idx in UPPER_BODY_LMS.keys():
        lm = results.pose_landmarks.landmark[idx]
        vals += [lm.x, lm.y, lm.z, lm.visibility]

    return vals

# ================================================================
# Prediction from Sequence (BiLSTM + SVM)
# ================================================================
def predict(model, svm, scaler, seq_np):
    # Normalize using DataFrame with correct feature names
    flat = seq_np.reshape(-1, FEATURE_DIM)
    df = pd.DataFrame(flat, columns=COL_ORDER)
    scaled = scaler.transform(df)
    scaled = scaled.reshape(seq_np.shape)

    x = torch.tensor(scaled, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits, emb = model(x)
        emb_np = emb.cpu().numpy()

        probs_bi = torch.softmax(logits, dim=1).cpu().numpy()[0]

        try:
            probs_svm = svm.predict_proba(emb_np)[0]
        except:
            df = svm.decision_function(emb_np)
            exp = np.exp(df - np.max(df))
            probs_svm = exp / exp.sum()

        probs = ALPHA * probs_bi + (1-ALPHA) * probs_svm
        pred = int(np.argmax(probs))

    return pred, probs, probs_bi, probs_svm, emb_np

# ================================================================
# RL State
# ================================================================
def build_state(embedding, label, confidence, bad_history):
    emb = embedding.flatten()
    recent_bad = float(sum(bad_history)/len(bad_history)) if len(bad_history) else 0.0
    return np.concatenate([emb, np.array([label, recent_bad, confidence], dtype=np.float32)])

# ================================================================
# UI: Split Screen Renderer
# ================================================================
def draw_rl_panel(panel, action):
    cv2.putText(panel, "ErgoSense RL Actions", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.putText(panel, "Current Action:", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    y = 130
    if action == 0:
        cv2.putText(panel, "No Action", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 2)

    elif action == 1:
        cv2.putText(panel, "Soft Reminder:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(panel, "Sit upright / align spine", (20, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    elif action == 2:
        cv2.putText(panel, "Strong Reminder!", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(panel, "Straighten immediately", (20, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    elif action == 3:
        cv2.putText(panel, "Specific Advice:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(panel, "Raise monitor / align neck", (20, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)


# ================================================================
# Main
# ================================================================
def main():
    model, svm, scaler, rl_agent, emb_dim = load_models()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    buffer = deque(maxlen=SEQ_LEN)
    bad_history = deque(maxlen=30)
    pred_history = deque(maxlen=5)

    last_action_time = 0
    COOLDOWN = 3.0

    rl_log = os.path.join(LOG_DIR, "rl_memory.csv")

    if not os.path.exists(rl_log):
        with open(rl_log, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "label", "confidence", "action", "recent_bad"])

    print("Starting ErgoSense demo...")

    current_action = 0
    action_expire_time = 0
    ACTION_HOLD_TIME = 3.0   # seconds to show the action
    RL_COOLDOWN = 2.0


    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # feature extraction
        features = extract_upper_body(results)
        buffer.append(features)

        action = 0
        posture_text = "Collecting..."

        if len(buffer) == SEQ_LEN:
            seq_np = np.array(buffer)
            pred, probs, probs_bi, probs_svm, emb_np = predict(model, svm, scaler, seq_np)

            pred_history.append(pred)
            final_pred = int(round(np.mean(pred_history)))
            confidence = probs[final_pred]

            bad_history.append(int(final_pred==1))

            posture_text = "GOOD POSTURE" if final_pred==0 else "BAD POSTURE"

            # -------------------------------------
            # 1. HARDCODED RULE: Only act on BAD posture
            # -------------------------------------
            if final_pred == 1 and confidence < 0.60:
                goto_rl = False    # posture uncertain → don't alert

            if final_pred == 0:          # GOOD POSTURE
                action = 0               # force NO action
                current_action = 0
                # Also reset cooldown timer so RL doesn’t fire immediately when switching
                last_action_time = time.time()
                action_expire_time = 0
                # Skip RL agent entirely for good posture
                goto_rl = False
            else:
                goto_rl = True




            # # RL action
            # if rl_agent and time.time() - last_action_time > COOLDOWN:
            #     state = build_state(emb_np, final_pred, confidence, bad_history)
            #     s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            #     with torch.no_grad():
            #         q = rl_agent(s_t)
            #         action = int(torch.argmax(q).item())
            #     last_action_time = time.time()

            #     # log RL experience
            #     with open(rl_log, "a", newline="") as f:
            #         writer = csv.writer(f)
            #         writer.writerow([
            #             time.time(),
            #             final_pred,
            #             float(confidence),
            #             action,
            #             sum(bad_history)/len(bad_history)
            #         ])


            if goto_rl:
            # --------------------------
            # RL ACTION SELECTION (STABLE)
            # --------------------------

                now = time.time()

                # If previous action is still active, keep showing it
                if now < action_expire_time:
                    action = current_action

                else:
                    # If cooldown passed, choose a new RL action
                    if rl_agent and (now - last_action_time) > RL_COOLDOWN:

                        # Build RL state vector
                        state = build_state(emb_np, final_pred, confidence, bad_history)
                        s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                        with torch.no_grad():
                            q_vals = rl_agent(s_t)
                            new_action = int(torch.argmax(q_vals).item())

                        # Lock the new action for next 3 seconds
                        current_action = new_action
                        action = new_action

                        last_action_time = now
                        action_expire_time = now + ACTION_HOLD_TIME

                        # Log RL experience
                        with open(rl_log, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                now, final_pred, float(confidence), new_action,
                                sum(bad_history)/len(bad_history)
                            ])
                    else:
                        # No new action: stay idle
                        action = 0
                        current_action = 0


        # -------------------------
        # Create RL panel
        # -------------------------
        h, w = frame.shape[:2]
        panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # dark bg
        draw_rl_panel(panel, action)

        # -------------------------
        # Combine frames
        # -------------------------
        combined = np.hstack((frame, panel))

        # Status bar
        cv2.putText(combined, posture_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,255,0) if posture_text=="GOOD POSTURE" else (0,0,255), 2)

        cv2.imshow("ErgoSense Live", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            buffer.clear()
            bad_history.clear()
            pred_history.clear()
            print("Buffers cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    