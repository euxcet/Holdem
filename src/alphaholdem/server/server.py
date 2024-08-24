from __future__ import annotations

import json
import socket
import struct
from threading import Thread
from .room import Room
from .client import Client
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

class Server():
    def __init__(
        self,
        address: str = '0.0.0.0',
        port: int = 19000,
    ) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((address, port))
        self.server.listen(10)
        accept_thread = Thread(target=self.accept_client)
        accept_thread.setDaemon(True)
        accept_thread.start()
        self.rooms = {}
        self.clients = set()
    
    def accept_client(self) -> None:
        while True:
            client_socket, client_info = self.server.accept()
            client_thread = Thread(target=self.handle_socket_client, args=(client_socket, client_info))
            client_thread.setDaemon(True)
            client_thread.start()

    def get_room(self, room_id: str, room_type: str = None) -> Room:
        if room_id not in self.rooms:
            room = Room(id=room_id)
            self.rooms[room_id] = room
        return self.rooms[room_id]

    def handle_socket_client(self, client_socket: socket.socket, client_info) -> None:
        client = Client(
            socket = client_socket,
            info = client_info,
            room = None,
        )
        self.handle_client(client)
    
    def handle_client(self, client: Client) -> None:
        self.clients.add(client)
        client.handle_event(event_type='info', message='Connected')
        while True:
            try:
                length = struct.unpack('i', client.socket.recv(4))[0]
                data = client.socket.recv(length).decode(encoding='utf-8')
                print(data)
                message = json.loads(data)
                if message['type'] == 'enter':
                    client.enter(self.get_room(message['room_id']))
                elif message['type'] == 'ready':
                    client.ready()
                elif message['type'] == 'leave':
                    client.leave()
                elif message['type'] == 'step':
                    client.step(message['action'])
            except Exception as e:
                print(e)
                client.leave(verbose=False)
                self.clients.remove(client)
                return
