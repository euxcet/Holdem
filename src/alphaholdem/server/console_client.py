import json
import socket
import struct

'''
Enter
    type: enter
    room_id:
    room_type:

Ready
    type: ready

Leave
    type: leave

Step
    type: step
    action:

'''
s = socket.socket()
def send(message):
    data = json.dumps(message)
    s.sendall(struct.pack('i', len(data)))
    s.sendall(bytes(data, encoding='utf-8'))

def main():
    s.connect(('0.0.0.0', 19000))

    send({'type': 'enter', 'room_id': '100'})
    send({'type': 'ready'})

    while True:
        data = s.recv(1024)
        print(data)
