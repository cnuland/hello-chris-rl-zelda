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
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Global data queue for SSE
data_queue = Queue()
latest_data = {
    'training': {},
    'vision': {}
}

# Simple session management - just track for display, accept all updates
session_lock = threading.Lock()


def register_session():
    """
    Register a new training session to the HUD.
    SIMPLIFIED: Always succeeds, just returns a session ID for tracking.
    All workers can send updates simultaneously!
    
    Returns:
        session_id: UUID for this session
    """
    new_session_id = str(uuid.uuid4())
    print(f"✅ HUD session registered: {new_session_id[:8]}... (no restrictions)")
    return new_session_id


def is_session_active(session_id):
    """
    Check if a session is active.
    SIMPLIFIED: Always returns True - all workers can send!
    
    Args:
        session_id: Session UUID to check
        
    Returns:
        bool: Always True
    """
    return True  # Accept updates from any worker!


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


@app.route('/api/register_session', methods=['POST'])
def api_register_session():
    """API endpoint to register a new training session"""
    session_id = register_session()
    if session_id:
        return jsonify({'session_id': session_id, 'status': 'registered'}), 200
    else:
        return jsonify({'error': 'Another session is active'}), 409


@app.route('/api/update_training', methods=['POST'])
def api_update_training():
    """API endpoint to update training data"""
    data = request.json
    session_id = data.get('session_id')
    training_data = data.get('data', {})
    
    if update_training_data(training_data, session_id):
        return jsonify({'status': 'updated'}), 200
    else:
        return jsonify({'error': 'Session not active'}), 403


@app.route('/api/update_vision', methods=['POST'])
def api_update_vision():
    """API endpoint to update vision data"""
    data = request.json
    session_id = data.get('session_id')
    image_base64 = data.get('image_base64')
    response_time = data.get('response_time')
    
    if update_vision_data(image_base64, response_time, session_id):
        return jsonify({'status': 'updated'}), 200
    else:
        return jsonify({'error': 'Session not active'}), 403


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Kubernetes"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/reset_session', methods=['POST'])
def api_reset_session():
    """
    API endpoint to force reset/clear the active session.
    Useful when restarting training jobs to clear stale sessions.
    """
    global active_session_id, active_session_timestamp
    
    with session_lock:
        old_session = active_session_id
        active_session_id = None
        active_session_timestamp = None
        
        if old_session:
            print(f"🔄 Session reset: Cleared session {old_session[:8]}...")
            return jsonify({'status': 'reset', 'previous_session': old_session[:8]}), 200
        else:
            print(f"🔄 Session reset: No active session to clear")
            return jsonify({'status': 'reset', 'previous_session': None}), 200


def update_training_data(data, session_id=None):
    """
    Update training data and push to all connected clients.
    Accepts data from any worker as long as a session is registered.
    
    Args:
        data: Dictionary containing training metrics
        session_id: Session UUID (optional, not validated)
    
    Returns:
        bool: True if update was accepted, False if no session active
    """
    global latest_data, active_session_id
    
    # Accept data from any worker as long as ANY session is registered
    # This allows distributed workers to all send data to the HUD
    if active_session_id is None:
        return False  # No session registered at all
    
    # MERGE data instead of replacing to preserve fields from different sources
    # Workers send: game state, location, health, LLM data
    # Callback sends: training metrics (steps, epoch, reward)
    if 'training' not in latest_data:
        latest_data['training'] = {}
    
    latest_data['training'].update(data)  # Merge instead of replace
    
    data_queue.put({
        'type': 'training',
        'data': latest_data['training']  # Send merged data
    })
    return True


def update_vision_data(image_base64, response_time=None, session_id=None):
    """
    Update vision image and push to all connected clients.
    Accepts data from any worker as long as a session is registered.
    
    Args:
        image_base64: Base64 encoded JPEG image
        response_time: LLM response time in milliseconds (optional)
        session_id: Session UUID (optional, not validated)
        
    Returns:
        bool: True if update was accepted, False if no session active
    """
    global latest_data, active_session_id
    
    # Accept data from any worker as long as ANY session is registered
    # This allows distributed workers to all send data to the HUD
    if active_session_id is None:
        return False  # No session registered at all
    
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

