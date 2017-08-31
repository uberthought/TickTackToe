from http.server import BaseHTTPRequestHandler, HTTPServer
import time

from test2 import Test

hostName = "ec2-13-58-202-229.us-east-2.compute.amazonaws.com"
hostPort = 9000

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):

        test = Test()
        test_result = test.run()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>Title.</title></head>", "utf-8"))
        self.wfile.write(bytes("<body><p>This is a test.</p>", "utf-8"))
        self.wfile.write(bytes(test_result, "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

try:
    myServer.serve_forever()
except KeyboardInterrupt:
    pass

myServer.server_close()
print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))
