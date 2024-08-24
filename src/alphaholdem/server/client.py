from __future__ import annotations
import json
import socket
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .room import Room

class Client():
    def __init__(
        self,
        socket: socket.socket,
        info,
        room: Room = None,
    ):
        self.socket = socket
        self.info = info
        self.room = room

    def handle_event(self, event_type: str, message) -> None:
        self.socket.sendall(bytes(json.dumps({'type': event_type, 'message': message}), encoding='utf-8'))

    def ready(self) -> None:
        if self.room is None:
            self.handle_event(event_type='error', message=f'You are not in any room.')
            return
        self.room.ready(self)

    def enter(self, room: Room) -> None:
        self.room = room
        room.enter(self)
        self.handle_event(event_type='info', message=f'Enter room {room.id}')
    
    def leave(self, verbose: bool = True) -> None:
        if self.room is None:
            if verbose:
                self.handle_event(event_type='error', message=f'You are not in any room.')
            return
        self.room.leave(self)
        self.room = None

    def step(self, action: str) -> None:
        if self.room is None:
            self.handle_event(event_type='error', message=f'You are not in any room.')
            return
        self.room.step(self, action)