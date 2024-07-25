from xmlrpc.server import SimpleXMLRPCServer
import threading
import time

def foo(n_times,string_list):
    """Replicates a string list n_times"""
    if not isinstance(n_times,int):
        raise ValueError("Invalid n_times parameter")
    if not isinstance(string_list,list):
        raise ValueError("Invalid string_list parameter")
    if not all(isinstance(s,str) for s in string_list):
        raise ValueError("Invalid string_list contents")
    return string_list*n_times

TIME_TO_RUN = 30.0
start_time = None
def done():
    global start_time
    if start_time is None:
        start_time = time.time()
    return time.time()-start_time > TIME_TO_RUN

PORT = 5678
server = SimpleXMLRPCServer(('127.0.0.1',PORT))
server.register_function(foo,"foo")   #advertise "foo"
server.register_introspection_functions()  #need this for listMethods to work

#the SimpleXMLRPCServer event loop is rather complex, so instead,
#the "done" checking is run in a separate thread.  If you don't want to
#kill the server with an internal signal, you can eliminate all this
#done_checker logic.
def check_for_done(server):
    while True:
        if done():
            print("Stopping server")
            server.shutdown()
            break
        time.sleep(0.1)

done_checker = threading.Thread(target=check_for_done,args=(server,))
done_checker.daemon = True   #this will kill the thread if the main thread is killed
done_checker.start()

print("Starting XML-RPC server on localhost:"+str(PORT))
server.serve_forever()
done_checker.join()