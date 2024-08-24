import time
from .server import Server

def start_server():
    Server()
    while True:
        time.sleep(1)