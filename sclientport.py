import os
import socket
import argparse

parser = argparse.ArgumentParser("This is client for multithreaded socket")
parser.add_argument('--host', metavar='host', type= str, nargs='?',default=socket.gethostname())
parser.add_argument('--port', metavar='port', type= int, nargs='?', default=9999)
args = parser.parse_args()

print(f"Connecting to server {args.host} on port: {args.port}")

with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sck:
    try:
        sck.connect((args.host,args.port))

    except Exception as e:
        raise SystemExit(f"We have failed to connect to host: {args.host} on port: {args.port}, because: {e}")

    while True:
        msg = input('What do we want to send to the server?: ')
        sck.sendall(msg.encode('utf-8'))
        if msg == 'exit':
            print('Client is saying goodbye')
            break
        data = sck.recv(1024)
        print(f"Server's response was {data.decode()}")

        

