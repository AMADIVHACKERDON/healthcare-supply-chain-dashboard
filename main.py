import socket

# Set up server A
server_a_host = '127.0.0.1'  # replace with actual IP address or hostname
server_a_port = 8000

server_a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_a_socket.bind((server_a_host, server_a_port))
server_a_socket.listen(1)

print(f"Server A listening on {server_a_host}:{server_a_port}")

# Set up server B
server_b_host = '127.0.0.1'  # replace with actual IP address or hostname
server_b_port = 9000

server_b_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_b_socket.bind((server_b_host, server_b_port))
server_b_socket.listen(1)

print(f"Server B listening on {server_b_host}:{server_b_port}")

# Connect server A to server B
server_b_address = (server_b_host, server_b_port)
server_a_socket.connect((server_b_address))

print(f"Server A connected to {server_b_host}:{server_b_port}")

# Send data from server A to server B
message = b"Hello, server B!"
server_a_socket.sendall(message)

# Receive data on server B
client_socket, client_address = server_b_socket.accept()
data = client_socket.recv(1024)

print(f"Received message from server A: {data.decode()}")

# Clean up
client_socket.close()
server_a_socket.close()
server_b_socket.close()
