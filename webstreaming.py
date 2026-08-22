# USAGE
# python webstreaming.py --input videos/city_flood.mp4 --ip 0.0.0.0 --port 8000

from flooddetector.prediction_on_video import PredictDisaster
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from collections import deque
import numpy as np
import threading
import argparse
import datetime
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-ip", "--ip", type=str, required=True,
	help="ip address of the device")
ap.add_argument("-o", "--port", type=int, required=True,
	help="ephemeral port number of the server (1024 to 65535)")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

Q = deque(maxlen=args["size"])

vs = cv2.VideoCapture(args["input"])

@app.route("/")
def index():
	return render_template("index.html")

def predict_frame():
    global vs, outputFrame, lock
	
    pd = PredictDisaster()
    while True:
        (grabbed, frame) = vs.read()
 
        if not grabbed:
            break
        
        frame_preds = pd.predict(frame)

        if frame_preds is not None:
            Q.append(frame_preds)
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = i

            text = "activity: {}".format(label)
            cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		        1.25, (0, 255, 0), 5)
        
        with lock:
            outputFrame = frame.copy()

def generate():
    global outputFrame, lock

    while True:
	    with lock:
		    if outputFrame is None:
			    continue

		    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

		    if not flag:
			    continue
        
	    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':

    t = threading.Thread(target=predict_frame)
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

vs.stop()


                


