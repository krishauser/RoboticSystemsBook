import socket
import threading
import time
from typing import Callable

# Let's implement a simple application that just handles messages by printing them
TIME_TO_RUN = 30.0

class MyServerLogic:
    def __init__(self):
        self.t_start = None
    def on_message(self, addr, msg):
        print("Server: Received chunk",msg,"from",addr,flush=True)
    def done(self):   # run for 30s
        if self.t_start is None:
            self.t_start = time.time()
        return time.time() - self.t_start > TIME_TO_RUN
    def reset(self):
        self.t_start = None

# The rest is communication code

def run_server(port : int, client_loop : Callable, done : Callable):
    """Boilerplate code.  client_handler is a function f(socket) that handles 
    all communication with the client.  done() returns True if the server should
    stop. 
    
    (for a clean shutdown, the client handler should also return if done() is True)"""
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serversocket.bind(('localhost', port))                       #accepts connections only from localhost
    #serversocket.bind((socket.gethostname(), port))    #use gethostname if you want to expose your server to the outside world
    while not done():
        # accept connections from outside -- this call blocks
        client_loop(serversocket)
    serversocket.close()

PORT = 5678
server_logic = MyServerLogic()
server_error = False

def client_loop(serversocket : socket.socket):
    N = 1000
    chunk,addr = serversocket.recvfrom(N)
    if len(chunk)==0:
        print("Server: Server socket broken",flush=True)
        return False
    server_logic.on_message(addr,chunk.decode('utf8'))
    return True

def server_reset():
    global server_comm_error
    server_logic.reset()
    server_comm_error = False

def server_start():
    """Handles all of the server running."""
    server_reset()
    run_server(PORT, client_loop, lambda: server_logic.done() or server_comm_error)

if __name__ == '__main__':
    server_start()