import socket

# create a socket object
def connect(host,port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
    remote_ip=socket.gethostbyname(host)
    print(remote_ip)

# connection to hostname on the port.
    s.connect((remote_ip, port))

# Receive no more than 1024 bytes
    tm = s.recv(1024)




    print("The time got from the server is %s" % tm.decode('ascii'))
    return s