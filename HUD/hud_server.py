"""
VLM Vision Hybrid HUD Server
Provides real-time dashboard for training visualization
Multi-session protection: Only one training session can update the HUD at a time
"""

import json
import time
import threading
import uuid
from queue import Queue, Empty
from flask import Flask, render_template, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Global data queue for SSE
data_queue = Queue()
latest_data = {
    'training': {},
    'vision': {}
}

# Session management
active_session_id = None
active_session_timestamp = None
session_lock = threading.Lock()
SESSION_TIMEOUT = 30  # seconds - if no updates for 30s, session is considered dead


def register_session():
    """
    Register a new training session to the HUD.
    Only one session can be active at a time.
    
    Returns:
        session_id: UUID for this session, or None if another session is active
    """
    global active_session_id, active_session_timestamp
    
    with session_lock:
        current_time = time.time()
        
        # Check if there's an active session that hasn't timed out
        if active_session_id is not None:
            if active_session_timestamp and (current_time - active_session_timestamp) < SESSION_TIMEOUT:
                # Another session is still active
                print(f"⚠️  HUD already connected to session: {active_session_id[:8]}...")
                print(f"   New session denied. Wait for timeout or stop existing session.")
                return None
            else:
                # Previous session timed out, can take over
                print(f"🔄 Previous session {active_session_id[:8]}... timed out")
        
        # Register new session
        new_session_id = str(uuid.uuid4())
        active_session_id = new_session_id
        active_session_timestamp = current_time
        
        print(f"✅ HUD session registered: {new_session_id[:8]}...")
        return new_session_id


def is_session_active(session_id):
    """
    Check if a session is the active one.
    
    Args:
        session_id: Session UUID to check
        
    Returns:
        bool: True if this session is active
    """
    global active_session_id, active_session_timestamp
    
    with session_lock:
        if active_session_id != session_id:
            return False
        
        # Update timestamp to keep session alive
        active_session_timestamp = time.time()
        return True


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')


@app.route('/stream')
def stream():
    """Server-Sent Events stream for real-time updates"""
    def event_stream():
        while True:
            try:
                # Get data from queue with timeout
                data = data_queue.get(timeout=1.0)
                
                # Format as SSE message
                if data['type'] == 'training':
                    yield f"event: training_update\ndata: {json.dumps(data['data'])}\n\n"
                elif data['type'] == 'vision':
                    yield f"event: vision_update\ndata: {json.dumps(data['data'])}\n\n"
                    
            except Empty:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"
            except GeneratorExit:
                break
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


def update_training_data(data, session_id=None):
    """
    Update training data and push to all connected clients.
    Only updates if session_id matches the active session.
    
    Args:
        data: Dictionary containing training metrics
        session_id: Session UUID (optional for backward compatibility)
    
    Returns:
        bool: True if update was accepted, False if rejected
    """
    global latest_data
    
    # If session_id is provided, validate it
    if session_id is not None:
        if not is_session_active(session_id):
            return False
    
    latest_data['training'] = data
    data_queue.put({
        'type': 'training',
        'data': data
    })
    return True


def update_vision_data(image_base64, response_time=None, session_id=None):
    """
    Update vision image and push to all connected clients.
    Only updates if session_id matches the active session.
    
    Args:
        image_base64: Base64 encoded JPEG image
        response_time: LLM response time in milliseconds (optional)
        session_id: Session UUID (optional for backward compatibility)
        
    Returns:
        bool: True if update was accepted, False if rejected
    """
    global latest_data
    
    # If session_id is provided, validate it
    if session_id is not None:
        if not is_session_active(session_id):
            return False
    
    vision_data = {
        'image': image_base64,
    }
    if response_time is not None:
        vision_data['response_time'] = response_time
    
    latest_data['vision'] = vision_data
    data_queue.put({
        'type': 'vision',
        'data': vision_data
    })
    return True


def start_server(host='0.0.0.0', port=8086):
    """
    Start the HUD server in a separate thread
    
    Args:
        host: Host to bind to (0.0.0.0 for all interfaces)
        port: Port to listen on (default 8086)
    """
    print(f"🎮 Starting VLM Vision Hybrid HUD Server")
    print(f"   URL: http://{host}:{port}")
    print(f"   Dashboard will open automatically in browser")
    
    # Run Flask in production mode (use waitress for production)
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=4)
    except ImportError:
        # Fallback to Flask development server
        print("   Note: Using Flask dev server (install waitress for production)")
        app.run(host=host, port=port, threaded=True, debug=False)


def start_server_thread(host='0.0.0.0', port=8086):
    """
    Start the HUD server in a background thread
    
    Args:
        host: Host to bind to
        port: Port to listen on
        
    Returns:
        Thread object running the server
    """
    server_thread = threading.Thread(
        target=start_server,
        args=(host, port),
        daemon=True
    )
    server_thread.start()
    
    # Give server time to start
    time.sleep(2)
    
    # Try to open browser
    try:
        import webbrowser
        webbrowser.open(f'http://localhost:{port}')
    except:
        pass
    
    return server_thread


if __name__ == '__main__':
    # For testing purposes
    start_server(host='0.0.0.0', port=8086)

