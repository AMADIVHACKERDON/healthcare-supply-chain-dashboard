import socket

def handle_client(conn_a, addr_a, conn_b, addr_b):
    # Receive message from client
    data = conn_a.recv(1024)

    # Pass message to server B
    conn_b.sendall(data)

    # Receive response from server B
    response = conn_b.recv(1024)

    # Send response back to client
    conn_a.sendall(response)

    # Clean up
    conn_a.close()
    conn_b.close()

def start_server(host_a, port_a, host_b, port_b):
    # Set up server A
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_a:
        sock_a.bind((host_a, port_a))
        sock_a.listen()

        print(f"Server A listening on {host_a}:{port_a}")

        while True:
            # Wait for incoming connection from client
            conn_a, addr_a = sock_a.accept()

            # Set up server B
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_b:
                sock_b.connect((host_b, port_b))

                print(f"Server A connected to Server B at {host_b}:{port_b}")

                # Handle client request
                handle_client(conn_a, addr_a, sock_b, (host_b, port_b))

if __name__ == '__main__':
    host_a = 'localhost'
    port_a = 8000
    host_b = 'localhost'
    port_b = 9000

    start_server(host_a, port_a, host_b, port_b)
