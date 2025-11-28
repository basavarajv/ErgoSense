# collect_data.py (ErgoSense Upper Body Version - Appending + Timestamp)
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Upper body landmarks and their names (11 keypoints)
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

def get_column_names():
    cols = []
    for _, name in UPPER_BODY_LANDMARKS.items():
        cols.extend([
            f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"
        ])
    cols.append("timestamp")
    return cols


def collect_posture_data(label, num_samples=300, save_dir='dataset'):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    data = []
    print(f"\n Collecting data for label: {label}")
    print("Press 'q' to quit early.")
    sample_count = 0

    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks for feedback
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
            )

            landmarks = []
            for idx in UPPER_BODY_LANDMARKS.keys():
                lm = results.pose_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            # Add timestamp (UNIX time)
            landmarks.append(time.time())
            data.append(landmarks)
            sample_count += 1

        cv2.putText(frame, f'Label: {label} ({sample_count}/{num_samples})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('ErgoSense Upper-Body Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save in append mode if file exists
    csv_path = os.path.join(save_dir, f'{label}.csv')
    columns = get_column_names()
    df = pd.DataFrame(data, columns=columns)

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f" Appended {len(data)} new samples to existing file: {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f" Created new dataset file with {len(data)} samples: {csv_path}")

    # Display total samples collected so far
    total_df = pd.read_csv(csv_path)
    print(f" Total samples for '{label}': {len(total_df)}")


if __name__ == "__main__":
    # Collect data for both classes
    collect_posture_data('good_posture', num_samples=500)
    collect_posture_data('bad_posture', num_samples=500)

    # Merge labeled data into one master dataset
    print("\n Merging good and bad posture data...")
    good_path = 'dataset/good_posture.csv'
    bad_path = 'dataset/bad_posture.csv'

    if os.path.exists(good_path) and os.path.exists(bad_path):
        good_df = pd.read_csv(good_path)
        bad_df = pd.read_csv(bad_path)

        good_df['label'] = 0
        bad_df['label'] = 1

        df = pd.concat([good_df, bad_df]).sample(frac=1).reset_index(drop=True)
        merged_path = 'dataset/posture_dataset.csv'
        df.to_csv(merged_path, index=False)

        print(f"Final merged dataset saved to {merged_path}")
        print(f"Total samples: {len(df)} | Columns: {len(df.columns)}")
    else:
        print("Warning: One or both class CSV files are missing. Skipping merge.")
