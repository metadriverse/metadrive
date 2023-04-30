"""

Example of using the render service to render a simple sphere.
The render service should be running.

"""

import sys
import pickle
import socket

# Address of the render service
renderer_ip = ("127.0.0.1", 62360)

# Port which the render service will ping when its done.
pingback_port = 62500

# Scene data
payload = pickle.dumps(
    {
        "scene": "resources/preview.bam",
        "dest": "demo-render.png",
        "view_size_x": 512,
        "view_size_y": 512,
        "pingback_port": pingback_port
    },
    protocol=0
)

# Start rendering
print("Sending payload ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.sendto(payload, renderer_ip)
except Exception as msg:
    sock.close()
    raise msg

# Wait until the renderer finished
print("Waiting for response ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
sock.settimeout(3.0)

try:
    sock.bind(("localhost", pingback_port))
except Exception as msg:
    print("Could not connect to pingback_port - maybe a render result is still waiting?")
    print("Error message:", msg)
    sys.exit(1)

sock.listen(1)

try:
    conn, addr = sock.accept()
except socket.timeout:
    print("No render pipeline response within timeout! Make sure the service is running.")
    sock.close()
    sys.exit(1)

print("Rendering was done!")
