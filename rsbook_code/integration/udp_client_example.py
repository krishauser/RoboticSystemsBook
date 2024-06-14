import socket
import threading
import time
from typing import Callable,Optional

def run_client(port : int, loop_func : Callable, done : Callable):
    """Boilerplate code.  loop_func is a function f(socket, addr) that handles 
    one iteration's communication with the server, and returns False on
    error.  done() returns True if the client should stop. 
    """
    print("Client: Connecting to port",port,flush=True)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ('localhost',port)
    while not done():
        if not loop_func(clientsocket, addr):
            break
    clientsocket.close()

#let's implement a simple handler and done function
import random
PORT = 5678
TIME_TO_RUN = 30.0
client_tstart = None
client_error = False
client_next_t_send = None
CLIENT_STRINGS = ['dog','cat','alligator','horse']

def client_done():
    """Run for 30s"""
    global client_tstart,client_error
    if client_error: return True
    if client_tstart is None:
        client_tstart = time.time()
    return time.time() - client_tstart > TIME_TO_RUN

def client_loop(clientsocket : socket.socket, addr):
    global client_error,client_next_t_send
    if client_next_t_send is None or time.time() >= client_next_t_send:
        msg = random.choice(CLIENT_STRINGS)
        print("Client: Sending",msg,flush=True)
        try:
            clientsocket.sendto(msg.encode('utf8'),addr)
        except Exception as e:
            print("Client: socket broken",flush=True)
            client_error = True 
            return False
        client_next_t_send = time.time() + 1.0
    return True

def client_reset():
    global client_tstart,client_error,client_next_t_send
    client_tstart = None
    client_error = False
    client_next_t_send = None

def client_start():
    """Handles all of the client running."""
    client_reset()
    run_client(PORT,client_loop,client_done)

if __name__ == '__main__':
    client_start()