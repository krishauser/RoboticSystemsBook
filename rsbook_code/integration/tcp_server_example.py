import socket
import threading
import time
from typing import Callable,Optional

def run_server(port : int, client_hander : Callable, done : Callable):
    """Boilerplate code.  client_handler is a function f(socket) that handles 
    all communication with the client.  done() returns True if the server should
    stop. 
    
    (for a clean shutdown, the client handler should also return if done() is True)"""
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('localhost', port))                       #accepts connections only from localhost
    #serversocket.bind((socket.gethostname(), port))    #use gethostname if you want to expose your server to the outside world
    # become a server socket
    serversocket.listen(1)                              #allow at most 1 client connection
    while not done():
        # accept connections from outside -- this call blocks
        (clientsocket, address) = serversocket.accept()
        print("Server: Accepted client from address",address,flush=True)
        # now start a thread to handle communication with the clientsocket
        ct = threading.Thread(target=client_hander,args=(clientsocket,))
        ct.daemon = True
        ct.run()
    serversocket.close()

#let's implement a simple handler and done function
PORT = 5678
TIME_TO_RUN = 30.0
server_tstart = None
server_error = False

def server_done():
    """Run for 30s"""
    global server_tstart,server_error
    if server_error: return True
    if server_tstart is None:
        server_tstart = time.time()
    return time.time() - server_tstart > TIME_TO_RUN

def client_loop(clientsocket : socket.socket):
    N = 1000
    chunk = clientsocket.recv(N)
    if len(chunk)==0:
        print("Server: Client socket broken",flush=True)
        return False
    print("Server: Received chunk",chunk.decode('utf8'),flush=True)
    return True

def handle_server_communication(clientsocket : socket.socket):
    global server_error
    while not server_done():
        if not client_loop(clientsocket):
            server_error = True
            break
    clientsocket.close()

def server_reset():
    global server_tstart,server_error
    server_tstart = None
    server_error = False

def server_start():
    """Handles all of the server running."""
    server_reset()
    run_server(PORT,handle_server_communication,server_done)

if __name__ == '__main__':
    server_start()