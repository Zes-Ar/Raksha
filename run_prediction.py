# USAGE
# python run_prediction.py --input videos/Assam_floods.mp4 --model flood
# python run_prediction.py --input videos/fire_video.mp4 --model fire

import tf_keras
from tf_keras.models import load_model
from collections import deque
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-m", "--model", type=str, default="flood", choices=["flood", "fire"],
    help="which model to use: flood or fire")
ap.add_argument("-s", "--size", type=int, default=64, help="queue size for averaging")
args = vars(ap.parse_args())

CLASSES = ["Flood", "Fire"]
model_path = os.path.join("output", f"{args['model']}.h5")

print(f"[INFO] Loading model from {model_path}...")
model = load_model(model_path)
print("[INFO] Model loaded successfully!")

Q = deque(maxlen=args["size"])

print(f"[INFO] Processing video: {args['input']}")
vs = cv2.VideoCapture(args["input"])
(W, H) = (None, None)

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    output = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224)).astype("float32")

    preds = model.predict(np.expand_dims(frame_resized, axis=0), verbose=0)[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    
    if i == 1:
        label = "Safe / No Disaster"
        color = (0, 255, 0)
    else:
        label = f"DISASTER: {CLASSES[0] if args['model'] == 'flood' else CLASSES[1]}"
        color = (0, 0, 255)

    confidence = results[i] * 100
    text = f"{label} ({confidence:.1f}%)"
    cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(output, f"Model: VGG-16 ({args['model']})", (20, H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("RAKSHA - Live Disaster Detection", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] Done.")
vs.release()
cv2.destroyAllWindows()
