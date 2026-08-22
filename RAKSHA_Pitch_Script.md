# RAKSHA — Pitch Script (4 Minutes)

## Screen: Landing Page (0:00 - 0:25)

"This is RAKSHA — Rapid Analysis and Knowledge-driven System for Hazard Awareness. An AI-powered disaster management platform that handles Detection, Alerts, and Relief routing — all in one system.

Every year India loses thousands of lives to floods, fires, and stampedes. The problem isn't just detection — it's the time gap between detecting a disaster and getting help to people. RAKSHA closes that gap using computer vision, intelligent routing, and real-time communication."

---

## Screen: Flood Detect — Select Guwahati (0:25 - 1:20)

"Starting with the Flood module. I'm selecting Guwahati, Assam — currently facing severe flooding. The system connects to CCTV nodes across the city, monitoring each feed in real time."

**[Click "Start ML Detection"]**

"Now our VGG-16 deep learning model is running inference on every frame — classifying it as 'Flood Detected' or 'Safe' with a confidence score. Predictions are averaged over 64 frames to eliminate false positives.

The model uses transfer learning — VGG-16 pre-trained on ImageNet, fine-tuned on our dataset of over 1,000 flood images. When flood is confirmed, the map highlights danger zones and triggers alerts automatically."

---

## Screen: Crowd Safety Detect (1:20 - 2:15)

"Now the Crowd Safety module — this is specifically designed to prevent stampedes. India has seen tragic stampedes at religious gatherings, railway stations, and festivals. Early detection of dangerous crowd density is the key to prevention."

**[Select city, show crowd feeds]**

"Here we use YOLOv3 — You Only Look Once version 3 — for real-time person detection. YOLO is an object detection model that processes entire frames in a single pass, making it fast enough for live video. It counts the number of people in each CCTV frame and tracks density violations.

When the crowd density crosses a threshold — say more than 10 people packed in a small zone — the system flags it as a stampede risk. The feed turns red, the node on the map changes, and an alert fires immediately.

The critical difference here is time. A stampede builds over 30-60 seconds before it becomes deadly. RAKSHA detects the dangerous density buildup in those early seconds — giving authorities a window to act: open exit gates, redirect foot traffic, or deploy police before it's too late."

---

## Screen: Alert Page (2:15 - 2:30)

"Alerts go out instantly — Email via SendGrid, SMS via Twilio — reaching all registered citizens and authorities in the affected zone. The alert includes disaster type, severity, and evacuation directions."

---

## Screen: Relief Page — Send Relief (2:30 - 3:15)

**[Click Send Relief, wait for routes]**

"Relief routing uses Google's Routes API to compute real road-following paths with live traffic data. These aren't straight lines — these are actual drivable routes with real ETAs.

The algorithm behind this is GRASP — Greedy Randomized Adaptive Search Procedure — a metaheuristic designed for Vehicle Routing Problems. Unlike simple shortest-path, GRASP handles multiple vehicles, multiple stops, and capacity limits simultaneously — which is exactly what disaster relief needs."

**[Click Animate]**

"These arrows represent vehicles dispatched along optimized routes. In production, these would track real NDRF vehicles via GPS."

---

## Tech Stack (3:15 - 3:35)

"Tech stack: Django backend, TensorFlow with VGG-16 for flood and fire classification, YOLOv3 for crowd person detection, Google Maps and Routes API for mapping and real-time routing, SendGrid and Twilio for dual-channel alerts, and MJPEG streaming for live video feeds. Everything runs on a single deployable server."

---

## Future Prospects & Unique Points (3:35 - 4:15)

"What's next for RAKSHA:

**Bluetooth Mesh Communication** — When cell towers go down in disasters, phones can form a peer-to-peer Bluetooth relay network. Alerts propagate through the crowd without cellular service — people become the network.

**Government CCTV Integration** — India has millions of cameras under Smart Cities Mission. RAKSHA plugs into existing infrastructure — no new hardware needed.

**Edge Deployment** — Running ML inference directly on CCTV nodes for millisecond-latency detection, even with intermittent connectivity.

**Multi-Disaster Correlation** — A flood triggers electrical fires, both cause crowd panic. RAKSHA handles all three simultaneously, giving responders a unified picture.

RAKSHA is not just a detection tool — it's a complete disaster response operating system. From the moment a camera sees rising water or a dangerous crowd buildup, to the moment rescue reaches the last person — every step is automated, optimized, and real-time.

Thank you."
