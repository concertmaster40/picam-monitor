# Web streaming and person detection using Picamera2 and OpenCV

import io
import logging
import smtplib
import socketserver
import threading
import time
from email.message import EmailMessage
from http import server
from threading import Condition

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# ----------------- CONFIGURATION -----------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "nickjackel@gmail.com"
SENDER_PASSWORD = "maxx rzda cytj xvat" 
RECIPIENT_EMAIL = "nickjackel@gmail.com"

# Minimum cooldown between email alerts in seconds
EMAIL_COOLDOWN = 30
# -------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)

def send_email_alert(image_bytes):
    """Sends an email with the captured frame attached."""
    try:
        msg = EmailMessage()
        msg["Subject"] = "Alert: Person Detected by Raspberry Pi"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        msg.set_content(
            f"A person was detected by your surveillance camera at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
            "See the attached image."
        )

        msg.add_attachment(
            image_bytes,
            maintype="image",
            subtype="jpeg",
            filename="person_detected.jpg"
        )

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

        logging.info("Email alert successfully sent.")
    except Exception as e:
        logging.error("Failed to send email alert: %s", e)

def person_detector_loop():
    """Background worker analyzing streaming frames for human figures."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    last_alert_time = 0

    while True:
        # Sample frames roughly 2 times per second to keep CPU load low on the Pi
        time.sleep(0.5)

        with output.condition:
            frame_bytes = output.frame

        if frame_bytes is None:
            continue

        # Convert JPEG buffer to OpenCV matrix
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Downscale for faster detection
        small_frame = cv2.resize(frame, (320, 240))

        # Detect pedestrians
        boxes, _ = hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        if len(boxes) > 0:
            current_time = time.time()
            if current_time - last_alert_time > EMAIL_COOLDOWN:
                logging.info("Person detected! Dispatching email alert...")
                last_alert_time = current_time
                # Dispatch email in a background thread so video streaming never stutters
                threading.Thread(
                    target=send_email_alert,
                    args=(frame_bytes,),
                    daemon=True
                ).start()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

output = StreamingOutput()

picam2 = Picamera2()
# Configuracao com rotacao de 180 graus (hflip=1, vflip=1) a nivel de hardware/ISP
video_config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    transform=Transform(hflip=1, vflip=1)
)
picam2.configure(video_config)
picam2.start_recording(JpegEncoder(), FileOutput(output))

# Start background person detection worker
detector_thread = threading.Thread(target=person_detector_loop, daemon=True)
detector_thread.start()

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    logging.info("Streaming server started on port 8000...")
    server.serve_forever()
finally:
    picam2.stop_recording()