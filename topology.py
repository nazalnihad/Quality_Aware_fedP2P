def get_neighbours(peer_id,num_peers,topology):
    if topology == "fully_connected":
        neighbours = [i for i in range(num_peers) if i!= peer_id]
        return neighbours
    elif topology == "ring":
        left = (peer_id - 1) % num_peers
        right = (peer_id + 1) % num_peers
        neighbours = list(set([left, right]))

        return neighbours

def get_peer_address(peer_id,base_port):
    return f"http://localhost:{base_port + peer_id}"
