from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer
from StreamData import StreamData

HOST = "0.0.0.0"
PORT = 9080


stream_data:dict[StreamData] = {
    "CamFull": StreamData("http://localhost:8080/stream/0"),
    "CamBL": StreamData("http://localhost:8080/stream/1"),
    "CamTL": StreamData("http://localhost:8080/stream/2"),
    "CamBR": StreamData("http://localhost:8080/stream/3"),
    "CamTR": StreamData("http://localhost:8080/stream/4"),
    # "0": StreamData("http://172.20.10.9:81/stream"),
    # "1": StreamData("http://172.20.10.8:81/stream"),
    # "2": StreamData("http://172.20.10.7:81/stream"),
}

# Main http server's stream handler
class StreamHandler(BaseHTTPRequestHandler):
    PRED_DIR = "/prediction/"
    RESTREAM_DIR = "/stream/"
    DIRECT_STREAM_DIR = "/direct/"

    def invalidAddress(self):
        self.wfile.write(b"Yeah you typed in the wrong address, sorry.\r\n")
        self.wfile.write(f"Try something like: \"http://IP{StreamHandler.PRED_DIR}X\", where X is the stream's ID\r\n".encode())
    
    def startStream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

    def sendJPEG(self, jpeg):
        # Write image data
        self.wfile.write(b"--frame\r\n")
        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
        self.wfile.write(jpeg)
        self.wfile.write(b"\r\n")

    def stream(self, stream_type:int, stream_id:str):
        # Validation of address
        stream:StreamData = None
        try:
            stream = stream_data[stream_id]
        except:
            self.invalidAddress()
            return
        # Address is valid after this point

        self.startStream()

        while True:
            # Get data based on stream type
            if stream_type == 1:
                res = stream.getResult()
                jpeg = res.getOutJPEG()
            elif stream_type == 2:
                res = stream.getResult()
                jpeg = res.getInJPEG()
            elif stream_type == 3:
                jpeg = stream.getJPEG()
            else:
                print(f"ERROR: Invalid stream_type: {stream_type}")
                return

            # Send stream data
            if jpeg is None:
                continue
            self.sendJPEG(jpeg)

    def do_GET(self):
        print(f"Client connecting at path {self.path}")
        stream_type = None
        stream_id = None

        if self.path[0:len(self.PRED_DIR)] == self.PRED_DIR: 
            stream_id = self.path[len(self.PRED_DIR):]
            stream_type = 1
        if self.path[0:len(self.RESTREAM_DIR)] == self.RESTREAM_DIR: 
            stream_id = self.path[len(self.RESTREAM_DIR):]
            stream_type = 2
        if self.path[0:len(self.DIRECT_STREAM_DIR)] == self.DIRECT_STREAM_DIR: 
            stream_id = self.path[len(self.DIRECT_STREAM_DIR):]
            stream_type = 3

        try:
            if stream_type is not None:
                self.stream(stream_type, stream_id)
            else:
                self.invalidAddress()
        except (BrokenPipeError, ConnectionResetError):
            print(f"Client disconnected from path {self.path}")
            pass

        print(f"Connection ended with client from path {self.path}")

        
    
    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    # Create server
    server = ThreadedHTTPServer((HOST,PORT), StreamHandler)

    # Each client gets a thread
    server.socket.setsockopt(__import__('socket').SOL_SOCKET, __import__('socket').SO_REUSEADDR, 1)
    print(f"Streaming at \"http://IP:{PORT}\"")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()



if __name__ == "__main__":
    main()