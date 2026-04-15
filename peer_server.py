from flask import Flask, request, jsonify
import threading
from utils import deserialize_model

app = Flask(__name__)

# Buffer: key = (round_num, peer_id) → value = state_dict
model_buffer = {}
buffer_lock = threading.Lock()

# Set by peer.py after startup
MY_DATA_SIZE = 0

@app.route('/receive_model', methods=['POST'])
def receive_model():
    data = request.get_json()
    peer_id = data['peer_id']
    round_num = data['round']
    state_dict = deserialize_model(data['state_dict'])
    with buffer_lock:
        model_buffer[(round_num, peer_id)] = state_dict
    return jsonify({"status": "success"})

@app.route('/peer_info', methods=['GET'])
def peer_info():
    """Expose this peer's data contribution size."""
    return jsonify({"data_size": MY_DATA_SIZE})

# Set by peer.py each round so richer peers can pull it
MY_LATEST_MODEL = None   # state_dict (serialized)
MY_LATEST_ROUND = -1

@app.route('/get_model', methods=['GET'])
def get_model():
    """Allow a richer peer to pull this peer's latest trained model."""
    if MY_LATEST_MODEL is None:
        return jsonify({"status": "not_ready"}), 503
    return jsonify({
        "status": "ok",
        "round": MY_LATEST_ROUND,
        "state_dict": MY_LATEST_MODEL,
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

def start_server(port):
    """Run Flask server in a background daemon thread."""
    thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, use_reloader=False),
        daemon=True
    )
    thread.start()