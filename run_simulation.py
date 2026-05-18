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

# Clear old done flags so we don't accidentally exit immediately
for pid in range(NUM_PEERS):
    done_file = f"outputs/peer_{pid}_done"
    if os.path.exists(done_file):
        os.remove(done_file)

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

# Wait for all to finish via done files, because peers must stay alive
# to serve their final models to slower peers.
while True:
    all_done = True
    for pid in range(NUM_PEERS):
        if not os.path.exists(f"outputs/peer_{pid}_done"):
            all_done = False
            break
    
    if all_done:
        break
        
    # Protection against a peer crashing and deadlocking the simulation
    for i, p in enumerate(processes):
        if p.poll() is not None:
            if not os.path.exists(f"outputs/peer_{i}_done"):
                print(f"\n[!] Error: Peer {i} crashed unexpectedly! Aborting simulation.")
                for p_to_kill in processes:
                    p_to_kill.terminate()
                sys.exit(1)
                
    time.sleep(2)

print("All peers completed successfully. Terminating processes...")
for p in processes:
    p.terminate()

total_time = time.time() - start_time
print("\n" + "="*60)
print(f"  Simulation complete! Total time: {total_time:.1f}s")
print(f"  Final models saved in: outputs/")
print("="*60)

# Print comparison table (reads stats JSONs saved by each peer — fast)
print("\nGenerating comparison table...")
subprocess.run([sys.executable, "compare.py"], cwd=os.path.dirname(os.path.abspath(__file__)))