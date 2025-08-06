import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import threading
from datetime import datetime
import json
import os
import tempfile
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="🚀 Spacecraft Emergency Control Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Control Room Look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .video-box {
        background: #f8f9fa;
        border: 2px solid #2a5298;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    
    .detection-alert {
        background: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    
    .safe-status {
        background: #00cc44;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    
    .warning-status {
        background: #ffaa00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    
    .debug-info {
        background: #e1f5fe;
        border: 1px solid #01579b;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 12px;
    }
    
    .video-header {
        background: #2a5298;
        color: white;
        padding: 5px 10px;
        border-radius: 5px 5px 0 0;
        margin: -10px -10px 10px -10px;
        font-weight: bold;
    }
    
    .feed-status {
        padding: 5px;
        border-radius: 5px;
        margin: 5px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .feed-active {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .feed-inactive {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'emergency_mode' not in st.session_state:
    st.session_state.emergency_mode = False
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = {'feed1': False, 'feed2': False, 'feed3': False}
if 'video_files' not in st.session_state:
    st.session_state.video_files = {'feed1': None, 'feed2': None, 'feed3': None}
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = {'feed1': False, 'feed2': False, 'feed3': False}

# Load Model with better error handling
@st.cache_resource
def load_detection_model(model_path=None):
    try:
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            return model, f"Custom model: {os.path.basename(model_path)}"
        else:
            model = YOLO('yolov8n.pt')
            return model, "YOLOv8n (default)"
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, f"Error: {str(e)}"

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 SPACECRAFT EMERGENCY CONTROL CENTER</h1>
    <h3>Multi-Feed Equipment Detection & Monitoring System</h3>
</div>
""", unsafe_allow_html=True)

# Load model
model, model_info = load_detection_model()

# Sidebar Control Panel
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Debug Mode Toggle
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
    # Model Upload Section
    st.subheader("🤖 AI Model Setup")
    uploaded_model = st.file_uploader(
        "Upload YOLO Model (.pt file):",
        type=['pt'],
        help="Upload your trained YOLO model"
    )
    
    if uploaded_model is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.read())
                model_path = tmp_file.name
            model, model_info = load_detection_model(model_path)
        except Exception as e:
            st.error(f"Error processing uploaded model: {str(e)}")
    
    st.success(f"📝 {model_info}")
    
    st.markdown("---")
    
    # Emergency Button
    if st.button("🚨 EMERGENCY ALERT 🚨", key="emergency_btn"):
        st.session_state.emergency_mode = not st.session_state.emergency_mode
        if st.session_state.emergency_mode:
            st.session_state.detection_log.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'event': 'EMERGENCY MODE ACTIVATED',
                'status': 'CRITICAL'
            })

    # Emergency Status
    if st.session_state.emergency_mode:
        st.markdown("""
        <div class="detection-alert">
        🚨 EMERGENCY MODE ACTIVE 🚨<br>
        All feeds monitored
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-status">
        ✅ NORMAL OPERATIONS
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Video Upload Section
    st.header("📹 Video Feeds")
    
    # Feed 1
    st.subheader("📹 Feed 1 - Main Camera")
    video_1 = st.file_uploader(
        "Upload Video for Feed 1:",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key="video_feed_1"
    )
    if video_1:
        st.session_state.video_files['feed1'] = video_1
        st.success(f"✅ Feed 1: {video_1.name}")
    
    # Feed 2
    st.subheader("📹 Feed 2 - Secondary Camera")
    video_2 = st.file_uploader(
        "Upload Video for Feed 2:",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key="video_feed_2"
    )
    if video_2:
        st.session_state.video_files['feed2'] = video_2
        st.success(f"✅ Feed 2: {video_2.name}")
    
    # Feed 3
    st.subheader("📹 Feed 3 - Emergency Camera")
    video_3 = st.file_uploader(
        "Upload Video for Feed 3:",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key="video_feed_3"
    )
    if video_3:
        st.session_state.video_files['feed3'] = video_3
        st.success(f"✅ Feed 3: {video_3.name}")
    
    st.markdown("---")
    
    # Detection Settings
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold:", 0.1, 0.95, 0.25, 0.05)
    detection_frequency = st.slider("Detection Frequency (fps):", 1, 10, 3)  # Reduced for stability
    
    # Global Controls
    st.subheader("🎛️ Global Controls")
    
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start All", type="primary"):
            for feed in ['feed1', 'feed2', 'feed3']:
                if st.session_state.video_files[feed]:
                    st.session_state.monitoring_active[feed] = True
    
    with col_stop:
        if st.button("⏹️ Stop All"):
            for feed in ['feed1', 'feed2', 'feed3']:
                st.session_state.monitoring_active[feed] = False
                st.session_state.video_processing[feed] = False

# Enhanced Detection Function
def detect_objects_in_frame(frame, feed_name):
    if model is None:
        return [], frame
    
    try:
        results = model(frame, conf=confidence_threshold, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    
                    if confidence > confidence_threshold:
                        if hasattr(model, 'names') and class_id in model.names:
                            class_name = model.names[class_id]
                        elif class_id < len(class_names):
                            class_name = class_names[class_id]
                        else:
                            class_name = f"Object_{class_id}"
                            
                        detections.append({
                            'equipment': class_name,
                            'confidence': confidence,
                            'feed': feed_name
                        })
            
            try:
                annotated_frame = result.plot()
            except:
                annotated_frame = frame
        else:
            annotated_frame = frame
        
        return detections, annotated_frame
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Detection error in {feed_name}: {str(e)}")
        return [], frame

# Process single video feed - FIXED VERSION
def process_video_feed(feed_name, video_file, video_container, detection_container):
    if not video_file or not st.session_state.monitoring_active[feed_name]:
        return
    
    # Prevent multiple processing instances
    if st.session_state.video_processing[feed_name]:
        return
    
    st.session_state.video_processing[feed_name] = True
    
    cap = None
    tfile_path = None
    
    try:
        # Create temporary file with proper cleanup
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        # Reset file pointer to beginning
        video_file.seek(0)
        video_bytes = video_file.read()
        
        if len(video_bytes) == 0:
            video_container.error(f"❌ {feed_name.upper()}: Empty video file")
            return
        
        tfile.write(video_bytes)
        tfile.close()
        tfile_path = tfile.name
        
        # Debug info
        if st.session_state.debug_mode:
            st.info(f"📁 {feed_name.upper()}: Processing file ({len(video_bytes)} bytes)")
        
        # Open video with error checking
        cap = cv2.VideoCapture(tfile_path)
        
        if not cap.isOpened():
            video_container.error(f"❌ {feed_name.upper()}: Cannot open video file")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        if st.session_state.debug_mode:
            detection_container.info(f"📹 {feed_name.upper()}: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        max_frames = min(100, total_frames) if total_frames > 0 else 100  # Limit frames for demo
        
        # Process frames
        while (cap.isOpened() and 
               st.session_state.monitoring_active[feed_name] and 
               frame_count < max_frames):
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (400, 300))
            
            # Add feed identifier
            cv2.putText(frame, f"{feed_name.upper()}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add emergency overlay if active
            if st.session_state.emergency_mode:
                cv2.putText(frame, "EMERGENCY", 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Detect objects
            detections, annotated_frame = detect_objects_in_frame(frame, feed_name)
            
            # Display frame
            video_container.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Display detection info
            if detections:
                detection_text = f"🎯 **{feed_name.upper()}** (Frame {frame_count}):\n"
                for det in detections:
                    confidence_emoji = "🟢" if det['confidence'] > 0.7 else "🟡"
                    detection_text += f"{confidence_emoji} {det['equipment']}: {det['confidence']:.1%}\n"
                    
                    # Log detection
                    st.session_state.detection_log.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'event': f"[{feed_name.upper()}] {det['equipment']} ({det['confidence']:.1%})",
                        'status': 'DETECTED'
                    })
                
                detection_container.success(detection_text)
            else:
                detection_container.info(f"🔍 {feed_name.upper()}: Frame {frame_count} - No objects detected")
            
            # Control frame rate
            time.sleep(1.0 / detection_frequency)
            
            # Break if stopped
            if not st.session_state.monitoring_active[feed_name]:
                break
        
        # Processing complete
        if frame_count >= max_frames:
            video_container.success(f"✅ {feed_name.upper()}: Processing complete ({frame_count} frames)")
        
    except Exception as e:
        error_msg = f"❌ {feed_name.upper()}: Error - {str(e)}"
        video_container.error(error_msg)
        if st.session_state.debug_mode:
            st.exception(e)
    
    finally:
        # Cleanup resources
        st.session_state.video_processing[feed_name] = False
        if cap is not None:
            cap.release()
        if tfile_path and os.path.exists(tfile_path):
            try:
                os.unlink(tfile_path)
            except:
                pass

# Main Content Area - Three Video Feeds
st.header("📺 Multi-Feed Monitoring Dashboard")

# Show system status
col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)
with col_sys1:
    model_status = "🟢 Loaded" if model else "🔴 Error"
    st.markdown(f"**Model:** {model_status}")
with col_sys2:
    uploaded_count = sum(1 for v in st.session_state.video_files.values() if v is not None)
    st.markdown(f"**Videos:** {uploaded_count}/3")
with col_sys3:
    active_count = sum(1 for v in st.session_state.monitoring_active.values() if v)
    st.markdown(f"**Active:** {active_count}/3")
with col_sys4:
    emergency_status = "🚨 ACTIVE" if st.session_state.emergency_mode else "✅ NORMAL"
    st.markdown(f"**Status:** {emergency_status}")

st.markdown("---")

# Create three columns for video feeds
col1, col2, col3 = st.columns(3)

# Feed 1
with col1:
    st.markdown("### 📹 FEED 1 - Main Camera")
    
    video_container_1 = st.empty()
    detection_info_1 = st.empty()
    
    # Status indicator
    if st.session_state.monitoring_active['feed1']:
        st.markdown('<div class="feed-status feed-active">🟢 ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="feed-status feed-inactive">🔴 INACTIVE</div>', unsafe_allow_html=True)
    
    # Control buttons for Feed 1
    col1a, col1b = st.columns(2)
    with col1a:
        if st.button("▶️ Start", key="start_feed1", 
                    disabled=not st.session_state.video_files['feed1'] or st.session_state.monitoring_active['feed1']):
            st.session_state.monitoring_active['feed1'] = True
            st.rerun()
    with col1b:
        if st.button("⏹️ Stop", key="stop_feed1"):
            st.session_state.monitoring_active['feed1'] = False
            st.session_state.video_processing['feed1'] = False
    
    # Process Feed 1
    if st.session_state.monitoring_active['feed1'] and st.session_state.video_files['feed1']:
        process_video_feed('feed1', st.session_state.video_files['feed1'], 
                          video_container_1, detection_info_1)
    else:
        # Show placeholder
        placeholder_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "FEED 1", 
                   (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if st.session_state.video_files['feed1']:
            cv2.putText(placeholder_frame, "Ready - Click Start", 
                       (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(placeholder_frame, "Upload Video First", 
                       (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        video_container_1.image(placeholder_frame, channels="BGR", use_container_width=True)
        detection_info_1.info("🔄 Feed 1 standby")

# Feed 2
with col2:
    st.markdown("### 📹 FEED 2 - Secondary Camera")
    
    video_container_2 = st.empty()
    detection_info_2 = st.empty()
    
    # Status indicator
    if st.session_state.monitoring_active['feed2']:
        st.markdown('<div class="feed-status feed-active">🟢 ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="feed-status feed-inactive">🔴 INACTIVE</div>', unsafe_allow_html=True)
    
    # Control buttons for Feed 2
    col2a, col2b = st.columns(2)
    with col2a:
        if st.button("▶️ Start", key="start_feed2", 
                    disabled=not st.session_state.video_files['feed2'] or st.session_state.monitoring_active['feed2']):
            st.session_state.monitoring_active['feed2'] = True
            st.rerun()
    with col2b:
        if st.button("⏹️ Stop", key="stop_feed2"):
            st.session_state.monitoring_active['feed2'] = False
            st.session_state.video_processing['feed2'] = False
    
    # Process Feed 2
    if st.session_state.monitoring_active['feed2'] and st.session_state.video_files['feed2']:
        process_video_feed('feed2', st.session_state.video_files['feed2'], 
                          video_container_2, detection_info_2)
    else:
        # Show placeholder
        placeholder_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "FEED 2", 
                   (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if st.session_state.video_files['feed2']:
            cv2.putText(placeholder_frame, "Ready - Click Start", 
                       (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(placeholder_frame, "Upload Video First", 
                       (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        video_container_2.image(placeholder_frame, channels="BGR", use_container_width=True)
        detection_info_2.info("🔄 Feed 2 standby")

# Feed 3
with col3:
    st.markdown("### 📹 FEED 3 - Emergency Camera")
    
    video_container_3 = st.empty()
    detection_info_3 = st.empty()
    
    # Status indicator
    if st.session_state.monitoring_active['feed3']:
        st.markdown('<div class="feed-status feed-active">🟢 ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="feed-status feed-inactive">🔴 INACTIVE</div>', unsafe_allow_html=True)
    
    # Control buttons for Feed 3
    col3a, col3b = st.columns(2)
    with col3a:
        if st.button("▶️ Start", key="start_feed3", 
                    disabled=not st.session_state.video_files['feed3'] or st.session_state.monitoring_active['feed3']):
            st.session_state.monitoring_active['feed3'] = True
            st.rerun()
    with col3b:
        if st.button("⏹️ Stop", key="stop_feed3"):
            st.session_state.monitoring_active['feed3'] = False
            st.session_state.video_processing['feed3'] = False
    
    # Process Feed 3
    if st.session_state.monitoring_active['feed3'] and st.session_state.video_files['feed3']:
        process_video_feed('feed3', st.session_state.video_files['feed3'], 
                          video_container_3, detection_info_3)
    else:
        # Show placeholder
        placeholder_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "FEED 3", 
                   (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if st.session_state.video_files['feed3']:
            cv2.putText(placeholder_frame, "Ready - Click Start", 
                       (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(placeholder_frame, "Upload Video First", 
                       (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        video_container_3.image(placeholder_frame, channels="BGR", use_container_width=True)
        detection_info_3.info("🔄 Feed 3 standby")

# Detection Log Section
st.markdown("---")
st.header("📝 Detection Log")

if st.session_state.detection_log:
    # Show recent detections
    for log_entry in st.session_state.detection_log[-10:]:
        if log_entry['status'] == 'CRITICAL':
            status_color = "detection-alert"
        else:
            status_color = "safe-status"
            
        st.markdown(f"""
        <div class="{status_color}">
        🕐 {log_entry['time']} - {log_entry['event']}
        </div>
        """, unsafe_allow_html=True)
    
    # Clear log button
    if st.button("🗑️ Clear Log"):
        st.session_state.detection_log = []
        st.rerun()
else:
    st.info("No detections logged yet. Start monitoring to see results.")

# Footer with metrics
st.markdown("---")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    st.metric("Model", "Loaded" if model else "Error", model_info[:15] + "..." if len(model_info) > 15 else model_info)
with col_f2:
    st.metric("Active Feeds", f"{active_count}/3", "monitoring")
with col_f3:
    st.metric("Total Detections", len(st.session_state.detection_log), "objects found")
with col_f4:
    emergency_count = len([log for log in st.session_state.detection_log if log['status'] == 'CRITICAL'])
    st.metric("Emergency Events", emergency_count, "critical alerts")
