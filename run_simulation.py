import subprocess
import sys
import os
import time

from config import NUM_PEERS

print("="*60)
print("  P2P Federated Learning Simulation")
print(f"  Peers: {NUM_PEERS}")
print("="*60)

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

# Launch all peers
processes = []
start_time = time.time()

for peer_id in range(NUM_PEERS):
    print(f"Launching Peer {peer_id}...")
    p = subprocess.Popen(
        [sys.executable, "peer.py", "--peer_id", str(peer_id)],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    processes.append(p)

print(f"\nAll {NUM_PEERS} peers launched. Waiting for completion...\n")

# Wait for all to finish
for p in processes:
    p.wait()

total_time = time.time() - start_time
print("\n" + "="*60)
print(f"  Simulation complete! Total time: {total_time:.1f}s")
print(f"  Final models saved in: outputs/")
print("="*60)

# Print comparison table (reads stats JSONs saved by each peer — fast)
print("\nGenerating comparison table...")
subprocess.run([sys.executable, "compare.py"], cwd=os.path.dirname(os.path.abspath(__file__)))