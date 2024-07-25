import socket
import time
from typing import Callable

# Let's implement a simple application that samples strings
import random
TIME_TO_RUN = 30.0
CLIENT_STRINGS = ['dog','cat','alligator','horse']

class MyClientLogic:
    def __init__(self):
        self.t_start = None
    def rate(self):
        return 1.0   #runs at 1hz
    def update(self) -> str:
        if self.t_start is None:
            self.t_start = time.time()
        return random.choice(CLIENT_STRINGS)
    def done(self):   # run for 30s
        if self.t_start is None:
            self.t_start = time.time()
        return time.time() - self.t_start > TIME_TO_RUN
    def reset(self):
        self.t_start = None


# The rest is communication code

def run_client(port : int, comm_handler : Callable):
    """Boilerplate code.  comm_handler is a function f(socket) that handles 
    the communication loop. 
    """
    print("Client: Connecting to port",port,flush=True)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost',port))
    comm_handler(clientsocket)
    clientsocket.close()

PORT = 5678
client_logic = MyClientLogic()

def handle_client_communication(clientsocket : socket.socket):
    client_logic.reset()
    client_comm_error = False
    dt = 1.0/client_logic.rate()
    while not client_logic.done() and not client_comm_error:
        t0 = time.time()
        msg = client_logic.update()
        if msg is not None:
            print("Client: Sending",msg,flush=True)
            try:
                clientsocket.send(msg.encode('utf8'))
            except Exception as e:
                print("Client: socket broken",flush=True)
                client_comm_error = True 
        t1 = time.time()
        time.sleep(max(dt - (t1-t0),0))

def client_start():
    """Handles all of the client running."""
    run_client(PORT,handle_client_communication)

if __name__ == '__main__':
    client_start()