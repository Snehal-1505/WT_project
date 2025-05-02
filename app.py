from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from mediapipe.python.solutions.pose import PoseLandmark
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import atexit
import queue
from queue import Queue
import os
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash
import secrets
from flask import Response
import calendar
import matplotlib.pyplot as plt
from io import BytesIO
import base64

print(secrets.token_hex(16))



atexit.register(lambda: camera.release())

engine = pyttsx3.init()
speech_queue = queue.Queue()


engine.setProperty('rate', 145)    # slower speech
engine.setProperty('volume', 1.0)  # max volume


app = Flask(__name__)
app.secret_key = "your_secret_key"

load_dotenv()
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)
camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
last_feedback = ""
last_speech_time=0 
speech_cooldown = 3  # seconds

def speak_feedback(message):
    global last_feedback, last_speech_time
    current_time = time.time()

    if message != last_feedback and (current_time - last_speech_time) > speech_cooldown:
        last_feedback = message
        last_speech_time = current_time
        threading.Thread(target=_speak, args=(message,)).start()

speech_queue = queue.Queue()
engine = pyttsx3.init()
speak_lock = threading.Lock()  # Global lock

def _speak_loop():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speaking thread
threading.Thread(target=_speak_loop, daemon=True).start()

def speak(text):
    speech_queue.put(text)

    
def _speak(message):
    engine.say(message)
    engine.runAndWait()

def calculate_angle(a, b, c):
    """Calculate angle between three points (a, b, c)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_bicep_curl(landmarks):
    shoulder = [landmarks[11].x, landmarks[11].y]
    elbow = [landmarks[13].x, landmarks[13].y]
    wrist = [landmarks[15].x, landmarks[15].y]

    angle = calculate_angle(shoulder, elbow, wrist)
    correct = 40 < angle < 160
    return angle, correct, elbow, shoulder, wrist

def analyze_squat(landmarks):
    hip = [landmarks[23].x, landmarks[23].y]
    knee = [landmarks[25].x, landmarks[25].y]
    ankle = [landmarks[27].x, landmarks[27].y]

    angle = calculate_angle(hip, knee, ankle)
    correct = 80 < angle < 120
    return angle, correct, knee, hip, ankle

def analyze_shoulder_press(landmarks):
    elbow = [landmarks[13].x, landmarks[13].y]
    shoulder = [landmarks[11].x, landmarks[11].y]
    hip = [landmarks[23].x, landmarks[23].y]

    angle = calculate_angle(elbow, shoulder, hip)
    correct = 70 < angle < 110
    return angle, correct, shoulder, elbow, hip

def generate_frames():
    global video_writer, recording_started

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        success, frame = camera.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        feedback_messages = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            angles_to_check = [
                {
                    "points": [12, 14, 16],  # Right Elbow
                    "name": "Right Elbow",
                    "threshold": (40, 160)
                },
                {
                    "points": [11, 13, 15],  # Left Elbow
                    "name": "Left Elbow",
                    "threshold": (40, 160)
                },
                {
                    "points": [24, 26, 28],  # Right Knee
                    "name": "Right Knee",
                    "threshold": (80, 120)
                },
                {
                    "points": [23, 25, 27],  # Left Knee
                    "name": "Left Knee",
                    "threshold": (80, 120)
                }
            ]

            all_correct = True

            for angle_info in angles_to_check:
                p1_idx, p2_idx, p3_idx = angle_info["points"]
                name = angle_info["name"]
                min_angle, max_angle = angle_info["threshold"]

                p1 = [landmarks[p1_idx].x, landmarks[p1_idx].y]
                p2 = [landmarks[p2_idx].x, landmarks[p2_idx].y]
                p3 = [landmarks[p3_idx].x, landmarks[p3_idx].y]

                angle = calculate_angle(p1, p2, p3)
                correct = min_angle < angle < max_angle
                color = (0, 255, 0) if correct else (0, 0, 255)

                # Convert normalized coords to pixel
                p1 = tuple(np.multiply(p1, [w, h]).astype(int))
                p2 = tuple(np.multiply(p2, [w, h]).astype(int))
                p3 = tuple(np.multiply(p3, [w, h]).astype(int))

                # Draw angle lines
                cv2.line(frame, p1, p2, color, 2)
                cv2.line(frame, p2, p3, color, 2)

                # Show only angle value near the joint
                cv2.putText(frame, f"{int(angle)}°", (p2[0] + 10, p2[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if not correct:
                    all_correct = False
                    feedback_messages.append(f"Adjust {name}")

            # Voice feedback only, no extra on-screen text
            if all_correct:
                speak_feedback("Great form!")
            else:
                for msg in feedback_messages:
                    speak_feedback(msg)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def stream():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # process frame
    finally:
        cap.release()
        print("Camera released.")

# In-memory "database"
users = {}

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("index"))
    else:
        return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = mongo.db.users.find_one({"email": email})
        print("User from DB:", user)  # 👀 See user data

        if user:
            print("Stored password hash:", user["password"])
            print("Entered password:", password)
            if check_password_hash(user["password"], password):
                session["user"] = email
                return redirect(url_for("index"))
            else:
                print("Password hash check failed!")
        else:
            print("User not found!")

        return "Invalid email or password. Please try again."

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        phone = request.form["phone"]
        address = request.form["address"]
        gender = request.form["gender"]

        # Check if user already exists
        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user:
            flash("Email already registered")
            return redirect("/register")
        hashed_password = generate_password_hash(password)

        # Insert into database
        mongo.db.users.insert_one({
            "name": name,
            "email": email,
            "password": hashed_password,
            "phone": phone,
            "address": address,
            "gender": gender
        })

        flash("Registration successful!")
        return redirect(url_for('login'))  # 👈 This redirects to login page
    
    return render_template("register.html")

@app.route("/index")
def index():
    user = session.get("user")
    print("Index route accessed by:", user)
    try:
        return render_template("index.html", user=user)
    except Exception as e:
        return f"Template render error: {e}"

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/about_us")
def about_us():
    return render_template("about.html")

@app.route("/help")
def help():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("help.html")
@app.route("/contact")
def contact():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("contact.html")

@app.route("/live-exercise")
def live_exercise():
    return render_template("exercise.html")  # or stream video if needed

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@atexit.register
def cleanup():
    print("Releasing camera...")
    if camera.isOpened():
        camera.release()

@app.route('/shutdown')
def shutdown():
    global cap
    if cap:
        cap.release()
        print("Camera released.")
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Shutting down..."    

@app.route("/test-db")
def test_db():
    try:
        mongo.db.test_collection.insert_one({"msg": "Hello from Flask!"})
        return "MongoDB connection successful!"
    except Exception as e:
        return f"MongoDB connection failed: {e}"


@app.route("/show-users")
def show_users():
    users = list(mongo.db.users.find())
    output = ""
    for user in users:
        output += f"<p>Email: {user['email']}</p>"
    return output or "No users found!"

@app.route("/count-users")
def count_users():
    count = mongo.db.users.count_documents({})
    return f"Total users in DB: {count}"


@app.route("/delete-users")
def delete_users():
    mongo.db.users.delete_many({})
    return "All users deleted"
@app.route("/clear-users")
def clear_users():
    mongo.db.users.delete_many({})
    return "Cleared all users. Now register again with hashed passwords."
@atexit.register
def cleanup():
    if video_writer:
        video_writer.release()


@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    user_data = mongo.db.users.find_one({"email": session["user"]})
    
    if not user_data:
        return "User not found!"
    
    return render_template("profile.html", user=user_data)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)