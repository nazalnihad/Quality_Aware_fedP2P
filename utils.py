import torch
import io
import base64

def serialize_model(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def deserialize_model(encoded_str):
    decoded =  base64.b64decode(encoded_str.encode('utf-8'))
    buffer = io.BytesIO(decoded)
    return torch.load(buffer)