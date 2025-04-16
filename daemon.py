# daemon.py
import os
import socket
import threading
from gi.repository import GLib
from pydbus import SessionBus
from facerec import check

SOCKET_PATH = f"/run/user/{os.getuid()}/face.sock"

# Clean up old socket
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

# D-Bus Daemon
class Daemon(object):
    """
    <node>
        <interface name='org.linux.FaceRecognition'>
            <method name='CheckFace'>
                <arg type='i' name='attempts' direction='in'/>
                <arg type='s' name='response' direction='out'/>
            </method>
        </interface>
    </node>
    """
    def CheckFace(self, attempts=5):
        return check(attempts)

def socket_listener():
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)  # Only owner can access
    server.listen(1)
    print(f"Socket server listening on {SOCKET_PATH}")
    while True:
        conn, _ = server.accept()
        with conn:
            try:
                data = conn.recv(32)
                if data.strip() == b'CheckFace':
                    result = check(5)
                    conn.sendall(result.encode('utf-8'))
                else:
                    conn.sendall(b"error")
            except Exception as e:
                print(f"Socket error: {e}")
                conn.sendall(b"fail")
try:
    # Launch socket listener thread
    threading.Thread(target=socket_listener, daemon=True).start()

    # D-Bus
    loop = GLib.MainLoop()
    bus = SessionBus()
    bus.publish("org.linux.Face", Daemon())
    print("Daemon running. Ctrl+C to stop.")
    loop.run()
except KeyboardInterrupt:
    print("Stopping...")

