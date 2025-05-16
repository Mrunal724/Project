 from flask import Flask, render_template, Response, jsonify, request, redirect, send_file, send_from_directory, url_for, flash
import cv2
import dlib
import numpy as np
import os
import logging
import mysql.connector
from mysql.connector import Error
from datetime import datetime, time as datetime_time
from threading import Lock
import time
import csv
import io
import uuid
import base64
import pandas as pd
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
import json
from functools import wraps
from dotenv import load_dotenv
import threading
from flask_sqlalchemy import SQLAlchemy

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FaceAttendanceSystem')

app = Flask(_name_)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key')
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['UNKNOWN_FACES_DIR'] = 'unknown_faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
mail = Mail(app)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize camera variable
camera = None
camera_lock = threading.Lock()
camera_active = False

def initialize_camera(camera_type='local'):
    global camera
    try:
        # First try to release any existing camera
        if camera is not None:
            camera.release()
            camera = None
            time.sleep(1)  # Give time for camera to be released

        if camera_type == 'ip':
            # Try to open IP camera
            try:
                logger.info("Attempting to connect to IP camera")
                # Try different IP camera URLs
                ip_urls = [
                    'http://192.0.0.4:8080/video',
                    'http://192.168.1.100:8080/video',
                    'http://192.168.0.100:8080/video'
                ]

                for url in ip_urls:
                    try:
                        logger.info(f"Trying IP camera URL: {url}")
                        # Use FFMPEG backend for IP camera
                        camera = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                        if camera.isOpened():
                            # Set IP camera properties
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            camera.set(cv2.CAP_PROP_FPS, 25)
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 550)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
                            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                            # Try to read a frame
                            ret, frame = camera.read()
                            if ret and frame is not None:
                                logger.info(f"Successfully connected to IP camera at {url}")
                                return True
                            else:
                                logger.warning(f"IP camera opened but failed to read frame from {url}")
                                camera.release()
                                camera = None
                    except Exception as e:
                        logger.warning(f"Failed to connect to IP camera at {url}: {str(e)}")
                        if camera is not None:
                            camera.release()
                            camera = None

                logger.error("Failed to connect to any IP camera URL")
                return False

            except Exception as e:
                logger.error(f"IP camera initialization error: {str(e)}")
                if camera is not None:
                    camera.release()
                    camera = None
                return False
        else:
            # Try to open local camera with different backends
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Default")
            ]

            for backend, backend_name in backends:
                for camera_index in [0, 1, 2]:
                    try:
                        logger.info(f"Trying camera index {camera_index} with {backend_name} backend")
                        camera = cv2.VideoCapture(camera_index, backend)

                        if camera.isOpened():
                            # Set camera properties
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            camera.set(cv2.CAP_PROP_FPS, 25)
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 550)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
                            camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

                            # Try to read a frame
                            ret, frame = camera.read()
                            if ret and frame is not None:
                                logger.info(f"Successfully initialized camera {camera_index} with {backend_name} backend")
                                return True
                            else:
                                logger.warning(f"Camera {camera_index} opened but failed to read frame with {backend_name}")
                                camera.release()
                                camera = None
                    except Exception as e:
                        logger.warning(f"Failed to open camera {camera_index} with {backend_name}: {str(e)}")
                        if camera is not None:
                            camera.release()
                            camera = None
            return False
    except Exception as e:
        logger.error(f"Camera initialization error: {str(e)}")
        if camera is not None:
            camera.release()
            camera = None
        return False

# ----------------------
# Face Recognition System
# ----------------------
class FaceRecognitionSystem:
    def _init_(self):
        self.model_dir = 'model'
        self.dataset_dir = 'dataset'
        self.threshold = 0.45
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(self.model_dir, 'shape_predictor_68_face_landmarks.dat'))
        self.recognizer = dlib.face_recognition_model_v1(os.path.join(self.model_dir, 'dlib_face_recognition_resnet_model_v1.dat'))
        self.known_embeddings = {}
        self.load_known_faces()
        self.load_config()

    def load_known_faces(self):
        self.known_embeddings = {}
        try:
            if not os.path.isdir(self.dataset_dir):
                logger.warning(f"Dataset directory {self.dataset_dir} does not exist.")
                return
            for person in os.listdir(self.dataset_dir):
                person_dir = os.path.join(self.dataset_dir, person)
                if os.path.isdir(person_dir):
                    embeddings = []
                    for img_file in os.listdir(person_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_dir, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                faces = self.detector(gray)
                                for face in faces:
                                    shape = self.predictor(img, face)
                                    embedding = np.array(self.recognizer.compute_face_descriptor(img, shape))
                                    embeddings.append(embedding)
                    if embeddings:
                        self.known_embeddings[person] = np.mean(embeddings, axis=0)
            logger.info(f"Loaded {len(self.known_embeddings)} known faces")
        except Exception as e:
            logger.error(f"Failed loading known faces: {str(e)}")

    def load_config(self):
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT config_value FROM system_config WHERE config_key = "face_recognition"')
            config = cursor.fetchone()
            if config:
                config_data = json.loads(config['config_value'])
                self.threshold = config_data.get('threshold', 0.45)
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")

    def process_frame(self, frame):
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.detector(gray)
            current_time = datetime.now()
            active_class = self.get_active_class(current_time)

            # Process each detected face
            for face in faces:
                try:
                    # Get facial landmarks
                    shape = self.predictor(frame, face)

                    # Compute face descriptor
                    embedding = np.array(self.recognizer.compute_face_descriptor(frame, shape))

                    # Recognize face
                    identity, distance = self.recognize_face(embedding)

                    # Handle unknown faces and attendance
                    if identity == "Unknown":
                        self.save_unknown_face(frame, face)
                    else:
                        self.update_attendances(identity, active_class, current_time)

                    # Draw annotation
                    self.draw_annotation(frame, face, identity, distance)

                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}")
                    continue

            return frame
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return frame

    def recognize_face(self, embedding):
        try:
            min_distance = float('inf')
            identity = "Unknown"
            for name, known_embed in self.known_embeddings.items():
                distance = np.linalg.norm(embedding - known_embed)
                if distance < self.threshold and distance < min_distance:
                    min_distance = distance
                    identity = name
            return identity, min_distance
        except Exception as e:
            logger.error(f"Recognition error: {str(e)}")
            return "Unknown", float('inf')

    def get_active_class(self, current_time):
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('''
                SELECT * FROM timetable
                WHERE class_date = %s
                AND start_time <= %s
                AND end_time >= %s
            ''', (current_time.date(), current_time.time(), current_time.time()))
            active_class = cursor.fetchone()
            cursor.close()
            conn.close()
            return active_class
        except Exception as e:
            logger.error(f"Active class query failed: {str(e)}")
            return None

    def update_attendances(self, name, active_class, timestamp):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Use the subject from active_class if available; otherwise, set as empty string
            subject = active_class['subject'] if active_class else ''

            cursor.execute('''
                INSERT INTO attendance (name, subject, status, timestamp)
                VALUES (%s, %s, 'Present', %s)
                ON DUPLICATE KEY UPDATE status='Present', timestamp=%s
            ''', (name, subject, timestamp, timestamp))

            if active_class:
                cursor.execute('''
                    INSERT INTO subject_attendance (timetable_id, student_name, status, timestamp)
                    VALUES (%s, %s, 'Present', %s)
                    ON DUPLICATE KEY UPDATE status='Present', timestamp=%s
                ''', (active_class['id'], name, timestamp, timestamp))

            conn.commit()

            # Send email notification
            if active_class:
                self.send_attendance_notification(name, active_class['subject'])

            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Attendance update failed: {str(e)}")
            if conn:
                conn.rollback()

    def save_unknown_face(self, frame, face):
        try:
            unknown_dir = app.config['UNKNOWN_FACES_DIR']
            os.makedirs(unknown_dir, exist_ok=True)
            # Ensure the face  are within image boundaries
            x1 = max(face.left(), 0)
            y1 = max(face.top(), 0)
            x2 = min(face.right(), frame.shape[1])
            y2 = min(face.bottom(), frame.shape[0])
            cropped_face = frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = os.path.join(unknown_dir, f'unknown_{timestamp}.jpg')
            cv2.imwrite(filename, cropped_face)
            logger.info(f"Saved unknown face: {filename}")
        except Exception as e:
            logger.error(f"Failed to save unknown face: {str(e)}")

    @staticmethod
    def draw_annotation(frame, face, identity, distance):
        try:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{identity} ({distance:.2f})" if identity != "Unknown" else "Unknown"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Annotation error: {str(e)}")

    def send_attendance_notification(self, student_name, subject):
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT email FROM users WHERE name = %s', (student_name,))
            user = cursor.fetchone()
            if user:
                msg = Message(
                    'Attendance Recorded',
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[user['email']]
                )
                msg.body = f'''
                Dear {student_name},

                Your attendance has been recorded for {subject} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

                Best regards,
                Attendance System
                '''
                mail.send(msg)
                logger.info(f"Attendance notification sent to {student_name} for {subject}")
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

face_system = FaceRecognitionSystem()

# ----------------------
# Camera Streaming Control
# ----------------------
def generate_frames():
    global camera, camera_active
    frame_count = 0
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    last_frame_time = time.time()
    min_frame_interval = 1.0 / 25.0  # 25 FPS
    last_successful_frame = time.time()
    max_time_without_frame = 5.0

    try:
        while camera_active:
            current_time = time.time()

            # Check if we need to reconnect
            if camera is None or not camera.isOpened() or (current_time - last_successful_frame > max_time_without_frame):
                logger.info("Camera needs reconnection")
                if reconnect_attempts < max_reconnect_attempts:
                    if initialize_camera():
                        reconnect_attempts = 0
                        last_successful_frame = current_time
                        continue
                    else:
                        reconnect_attempts += 1
                        logger.warning(f"Reconnection attempt {reconnect_attempts} failed")
                        time.sleep(2)
                        continue
                else:
                    logger.error("Max reconnection attempts reached")
                    break

            try:
                with camera_lock:
                    success, frame = camera.read()
                    if not success or frame is None:
                        logger.warning("Failed to read frame")
                        camera.release()
                        camera = None
                        time.sleep(1)
                        continue

                    # Update timestamps on successful frame read
                    last_frame_time = current_time
                    last_successful_frame = current_time
                    reconnect_attempts = 0

                    # Process every 4th frame
                    frame_count += 1
                    if frame_count % 4 == 0:
                        try:
                            frame = cv2.resize(frame, (550, 550))
                            processed_frame = face_system.process_frame(frame)
                        except Exception as e:
                            logger.error(f"Frame processing error: {str(e)}")
                            processed_frame = frame
                    else:
                        processed_frame = frame

                    # Encode frame
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ret:
                        logger.warning("Failed to encode frame")
                        continue

                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                continue

            # Ensure we don't process frames too quickly
            elapsed = time.time() - last_frame_time
            if elapsed < min_frame_interval:
                time.sleep(min_frame_interval - elapsed)

    except Exception as e:
        logger.error(f"Frame generation error: {str(e)}")
    finally:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
            camera_active = False

@app.route('/video_feed')
def video_feed():
    global camera_active
    if not camera_active:
        return Response(status=204)  # No content response if camera is not active
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def camera_control():
    global camera, camera_active
    try:
        data = request.get_json()
        action = data.get('action')
        camera_type = data.get('type', 'local')  # Default to local camera

        if action == 'start':
            with camera_lock:
                # Always release existing camera before starting new one
                if camera is not None:
                    camera.release()
                    camera = None
                    time.sleep(1)  # Give time for camera to be released

                try:
                    # Initialize camera with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        logger.info(f"Camera initialization attempt {attempt + 1} for {camera_type} camera")
                        if initialize_camera(camera_type):
                            camera_active = True
                            return jsonify({'status': 'success', 'message': f'{camera_type.capitalize()} camera started'})
                        time.sleep(1)
                    raise Exception(f"Failed to initialize {camera_type} camera after multiple attempts")
                except Exception as e:
                    logger.error(f"Camera initialization error: {str(e)}")
                    if camera is not None:
                        camera.release()
                        camera = None
                    raise Exception(f"Failed to initialize camera: {str(e)}")

        elif action == 'stop':
            with camera_lock:
                camera_active = False
                if camera is not None:
                    camera.release()
                    camera = None
                return jsonify({'status': 'success', 'message': 'Camera stopped'})

        return jsonify({'status': 'error', 'message': 'Invalid action'})

    except Exception as e:
        logger.error(f"Camera control error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/test_camera')
def test_camera():
    try:
        camera_type = request.args.get('type', 'local')
        # Try to initialize camera
        if initialize_camera(camera_type):
            if camera is not None:
                camera.release()
                camera = None
            return jsonify({'status': 'success', 'message': f'{camera_type.capitalize()} camera test successful'})
        else:
            return jsonify({'status': 'error', 'message': f'{camera_type.capitalize()} camera test failed'}), 500
    except Exception as e:
        logger.error(f"Camera test error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add a route to get available cameras
@app.route('/available_cameras')
def available_cameras():
    try:
        cameras = []
        # Test local cameras
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append({
                        'type': 'local',
                        'index': i,
                        'name': f'Local Camera {i}'
                    })
                cap.release()

        # Test IP cameras
        ip_urls = [
            'http://192.0.0.4:8080/video',
            'http://192.168.1.100:8080/video',
            'http://192.168.0.100:8080/video'
        ]

        for url in ip_urls:
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cameras.append({
                            'type': 'ip',
                            'url': url,
                            'name': f'IP Camera ({url})'
                        })
                cap.release()
            except:
                continue

        return jsonify({
            'status': 'success',
            'cameras': cameras
        })
    except Exception as e:
        logger.error(f"Error getting available cameras: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ----------------------
# Timetable Routes
# ----------------------
@app.route('/timetable')
def timetable():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT id, subject,
                   DATE_FORMAT(class_date, '%Y-%m-%d') as class_date,
                   TIME_FORMAT(start_time, '%H:%i') as start_time,
                   TIME_FORMAT(end_time, '%H:%i') as end_time
            FROM timetable
            ORDER BY class_date DESC, start_time ASC
        ''')
        timetable_records = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('timetable.html', timetable=timetable_records, readonly=False)
    except Exception as e:
        logger.error(f"Error fetching timetable: {str(e)}")
        return render_template('error.html', error="Failed to load timetable"), 500

@app.route('/manage_timetable', methods=['GET', 'POST'])
@app.route('/manage_timetable/<int:entry_id>', methods=['GET', 'POST'])
def manage_timetable(entry_id=None):
    entry = None
    conn = None
    cursor = None
    try:
        if entry_id:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM timetable WHERE id = %s', (entry_id,))
            entry = cursor.fetchone()
            cursor.close()
            conn.close()
        if request.method == 'POST':
            subject = request.form['subject'].strip()
            class_date = request.form['class_date']
            start_time = request.form['start_time']
            end_time = request.form['end_time']
            if datetime_time.fromisoformat(start_time) >= datetime_time.fromisoformat(end_time):
                raise ValueError("End time must be after start time")
            conn = get_db_connection()
            cursor = conn.cursor()
            if entry_id:
                cursor.execute('''
                    UPDATE timetable
                    SET subject = %s, class_date = %s, start_time = %s, end_time = %s
                    WHERE id = %s
                ''', (subject, class_date, start_time, end_time, entry_id))
            else:
                cursor.execute('''
                    INSERT INTO timetable (subject, class_date, start_time, end_time)
                    VALUES (%s, %s, %s, %s)
                ''', (subject, class_date, start_time, end_time))
            conn.commit()
            return redirect(url_for('timetable'))
        return render_template('manage_timetable.html', entry=entry)
    except Exception as e:
        logger.error(f"Timetable management error: {str(e)}")
        return render_template('error.html', error=str(e)), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/delete_timetable/<int:entry_id>')
def delete_timetable(entry_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM timetable WHERE id = %s', (entry_id,))
        conn.commit()
        return redirect(url_for('timetable'))
    except Exception as e:
        logger.error(f"Timetable deletion error: {str(e)}")
        return render_template('error.html', error="Deletion failed"), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ----------------------
# Attendance & Analytics Routes
# ----------------------

@app.route('/attendance', endpoint='attendance')
def attendance():
    try:
        # Get filter parameters from the request query string
        filter_date = request.args.get('date', '').strip()
        filter_subject = request.args.get('subject', '').strip()
        filter_name = request.args.get('name', '').strip()
        filter_status = request.args.get('status', '').strip()

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get total students
        cursor.execute('SELECT COUNT(DISTINCT name) as total FROM attendance')
        total = cursor.fetchone()
        total_students = total['total'] if total else 0

        # Get present students today
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(DISTINCT name) as present
            FROM attendance
            WHERE DATE(timestamp) = %s AND status = 'Present'
        ''', (today,))
        present = cursor.fetchone()
        present_students = present['present'] if present else 0

        # Calculate percentage
        percent_present = round((present_students / total_students * 100) if total_students > 0 else 0)

        #  query for attendance records with filtering.
        query = """
            SELECT name, subject, status,
                   DATE_FORMAT(timestamp, '%Y-%m-%d') as attendance_date,
                   TIME_FORMAT(timestamp, '%H:%i') as timestamp
            FROM attendance
            WHERE 1=1
        """
        params = []
        # Filter by date if provided
        if filter_date:
            query += " AND DATE(timestamp) = %s"
            params.append(filter_date)
        # Filter by subject if provided
        if filter_subject:
            query += " AND subject LIKE %s"
            params.append(f"%{filter_subject}%")
        # Filter by student name if provided
        if filter_name:
            query += " AND name LIKE %s"
            params.append(f"%{filter_name}%")
        # Filter by status if provided
        if filter_status:
            query += " AND status = %s"
            params.append(filter_status)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, tuple(params))
        attendance_records = cursor.fetchall()

        # Get unique subjects for filter
        cursor.execute('SELECT DISTINCT subject FROM attendance WHERE subject IS NOT NULL')
        subjects = [row['subject'] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        # Pass filter values back to template so they remain in the form.
        return render_template('attendance.html',
                             attendance_records=attendance_records,
                             total_students=total_students,
                             present_students=present_students,
                             percent_present=percent_present,
                             subjects=subjects,
                             filter_date=filter_date,
                             filter_subject=filter_subject,
                             filter_name=filter_name,
                             filter_status=filter_status)
    except Exception as e:
        logger.error(f"Error fetching attendance: {str(e)}")
        return render_template('error.html', error="Failed to load attendance"), 500

    try:
        # Get filter parameters from the request query string
        filter_date = request.args.get('date', '').strip()
        filter_subject = request.args.get('subject', '').strip()
        filter_name = request.args.get('name', '').strip()
        filter_status = request.args.get('status', '').strip()

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get total students
        cursor.execute('SELECT COUNT(DISTINCT name) as total FROM attendance')
        total = cursor.fetchone()
        total_students = total['total'] if total else 0

        # Get present students today
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(DISTINCT name) as present
            FROM attendance
            WHERE DATE(timestamp) = %s AND status = 'Present'
        ''', (today,))
        present = cursor.fetchone()
        present_students = present['present'] if present else 0

        # Calculate percentage
        percent_present = round((present_students / total_students * 100) if total_students > 0 else 0)

        # Build the query for attendance records with filtering.
        query = """
            SELECT name, subject, status,
                   DATE_FORMAT(%%Y-%%m-%%d') as attendance_date,
                   TIME_FORMAT(timestamp, '%%H:%%i') as timestamp
            FROM attendance
            WHERE 1=1
        """
        params = []
        # Filter by date if provided
        if filter_date:
            query += " AND DATE(timestamp) = %s"
            params.append(filter_date)
        # Filter by subject if provided
        if filter_subject:
            query += " AND subject LIKE %s"
            params.append(f"%{filter_subject}%")
        # Filter by student name if provided
        if filter_name:
            query += " AND name LIKE %s"
            params.append(f"%{filter_name}%")
        # Filter by status if provided
        if filter_status:
            query += " AND status = %s"
            params.append(filter_status)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, tuple(params))
        attendance_records = cursor.fetchall()

        # Get unique subjects for filter
        cursor.execute('SELECT DISTINCT subject FROM attendance WHERE subject IS NOT NULL')
        subjects = [row['subject'] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        # Pass filter values back to template so they remain in the form.
        return render_template('attendance.html',
                             attendance_records=attendance_records,
                             total_students=total_students,
                             present_students=present_students,
                             percent_present=percent_present,
                             subjects=subjects,
                             filter_date=filter_date,
                             filter_subject=filter_subject,
                             filter_name=filter_name,
                             filter_status=filter_status)
    except Exception as e:
        logger.error(f"Error fetching attendance: {str(e)}")
        return render_template('error.html', error="Failed to load attendance"), 500


@app.route('/download_attendance', endpoint='download_attendance_endpoint')
def download_attendance():
    try:
        # Retrieve filter parameters.
        filter_date = request.args.get('date', '').strip()
        filter_subject = request.args.get('subject', '').strip()
        filter_name = request.args.get('name', '').strip()

        conn = get_db_connection()
        query = """
            SELECT name, subject, status,
                   DATE_FORMAT(timestamp, '%%Y-%%m-%%d %%H:%%i:%%s') AS timestamp,
                   attendance_date
            FROM attendance
            ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Convert attendance_date to "YYYY-MM-DD" string.
        if 'attendance_date' in df.columns:
            df['attendance_date'] = pd.to_datetime(df['attendance_date']).dt.strftime('%Y-%m-%d')

        # Apply filters if provided.
        if filter_date:
            df = df[df['attendance_date'] == filter_date]
        if filter_subject:
            df = df[df['subject'].astype(str).str.contains(filter_subject, case=False, na=False)]
        if filter_name:
            df = df[df['name'].astype(str).str.contains(filter_name, case=False, na=False)]

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    except Exception as e:
        logger.exception("Error in download_attendance:")
        return render_template('error.html', error="Failed to download attendance: " + str(e)), 500

@app.route('/download_subject_attendance', endpoint='download_subject_attendance_endpoint')
def download_subject_attendance():
    try:
        filter_date = request.args.get('date', '').strip()
        filter_subject = request.args.get('subject', '').strip()
        filter_name = request.args.get('name', '').strip()

        conn = get_db_connection()
        query = """
            SELECT t.subject, t.class_date, sa.student_name, sa.status, sa.timestamp
            FROM subject_attendance sa
            JOIN timetable t ON sa.timetable_id = t.id
            ORDER BY t.class_date DESC, sa.timestamp DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if 'class_date' in df.columns:
            df['class_date'] = pd.to_datetime(df['class_date']).dt.strftime('%Y-%m-%d')

        if filter_date:
            df = df[df['class_date'] == filter_date]
        if filter_subject:
            df = df[df['subject'].astype(str).str.contains(filter_subject, case=False, na=False)]
        if filter_name:
            df = df[df['student_name'].astype(str).str.contains(filter_name, case=False, na=False)]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Subject Attendance")
        output.seek(0)
        return send_file(output,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True,
                         download_name=f"subject_attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    except Exception as e:
        logger.exception("Error in download_subject_attendance:")
        return render_template('error.html', error="Failed to download subject attendance: " + str(e)), 500

# ----------------------
# Registration & Unknown Faces
# ----------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form.get('name', '').strip()
            if not name:
                raise ValueError("Name is required")
            if 'file' in request.files:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], name)
                    os.makedirs(user_dir, exist_ok=True)
                    file.save(os.path.join(user_dir, filename))
            if 'captured_image' in request.form:
                image_data = request.form['captured_image'].split(',')[1]
                img_bytes = base64.b64decode(image_data)
                user_dir = os.path.join(app.config['UPLOAD_FOLDER'], name)
                os.makedirs(user_dir, exist_ok=True)
                filename = os.path.join(user_dir, f"{uuid.uuid4().hex}.jpg")
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
            face_system.load_known_faces()
            return jsonify(success=True, message="Registration successful")
        except Exception as e:
            logger.error(f"Registration failed: {str(e)}")
            return jsonify(success=False, message=str(e)), 400
    return render_template('register.html')

# Route to serve unknown faces files
@app.route('/unknown_faces/<path:filename>')
def unknown_face_file(filename):
    return send_from_directory(app.config['UNKNOWN_FACES_DIR'], filename)

@app.route('/unknown_faces')
def unknown_faces():
    try:
        unknown_dir = app.config['UNKNOWN_FACES_DIR']
        os.makedirs(unknown_dir, exist_ok=True)
        images = []
        for f in os.listdir(unknown_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get file creation time
                creation_time = os.path.getctime(os.path.join(unknown_dir, f))
                images.append({
                    'filename': f,
                    'timestamp': datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                })
        # Sort by timestamp, most recent first
        images.sort(key=lambda x: x['timestamp'], reverse=True)
        return render_template('unknown_faces.html', images=images)
    except Exception as e:
        logger.error(f"Error loading unknown faces: {str(e)}")
        return render_template('error.html', error="Failed to load unknown faces"), 500

# ----------------------
# Home / Dashboard Route
# ----------------------
@app.route('/')
def home():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get total students
        cursor.execute('SELECT COUNT(DISTINCT name) as total FROM attendance')
        total = cursor.fetchone()
        total_students = total['total'] if total else 0

        # Get present students today
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(DISTINCT name) as present
            FROM attendance
            WHERE DATE(timestamp) = %s AND status = 'Present'
        ''', (today,))
        present = cursor.fetchone()
        present_today = present['present'] if present else 0

        # Get active classes
        cursor.execute('''
            SELECT COUNT(*) as active
            FROM timetable
            WHERE class_date = %s
            AND start_time <= %s
            AND end_time >= %s
        ''', (today, datetime.now().time(), datetime.now().time()))
        active = cursor.fetchone()
        active_classes = active['active'] if active else 0

        # Calculate attendance rate
        attendance_rate = round((present_today / total_students * 100) if total_students > 0 else 0)

        # Get recent activities
        cursor.execute('''
            SELECT name as student_name,
                   DATE_FORMAT(timestamp, '%Y-%m-%d %H:%i:%s') as timestamp
            FROM attendance
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        recent_activities = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template('dashboard.html',
                             total_students=total_students,
                             present_today=present_today,
                             active_classes=active_classes,
                             attendance_rate=attendance_rate,
                             recent_activities=recent_activities)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return render_template('error.html', error="Failed to load dashboard"), 500

def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', 'Virendra@30'),
        database=os.environ.get('DB_NAME', 'attendance_system')
    )

if _name_ == '_main_':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
