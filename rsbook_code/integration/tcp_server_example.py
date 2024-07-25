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

PORT = 5678
server_logic = MyServerLogic()
server_comm_error = False
    
def handle_server_communication(clientsocket : socket.socket):
    global server_comm_error
    addr = clientsocket.getpeername()
    while not server_comm_error and not server_logic.done():
        N = 1000
        chunk = clientsocket.recv(N)  #this will need to be modified to properly handle framing
        if len(chunk)==0:
            print("Server: Client socket broken",flush=True)
            server_comm_error = True
            break
        else:
            server_logic.on_message(addr, chunk.decode('utf8'))
    clientsocket.close()

def server_reset():
    global server_comm_error
    server_logic.reset()
    server_comm_error = False

def server_start():
    """Handles all of the server running."""
    server_reset()
    run_server(PORT,handle_server_communication, lambda : server_logic.done() or server_comm_error)

if __name__ == '__main__':
    server_start()