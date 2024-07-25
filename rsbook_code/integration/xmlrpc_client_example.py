from xmlrpc.client import ServerProxy

PORT = 5678
server_connection = ServerProxy('http://127.0.0.1:'+str(PORT))

print("The server advertises:",server_connection.system.listMethods())

res = server_connection.foo(3,['bar','baz'])
print("The server returned",res)

print("Trying an incorrect call!")
res = server_connection.foo(3.5,['bar','baz'])
print("The server returned",res)