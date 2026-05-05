from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer
from StreamData import StreamData
from AllStreams import ALL_STREAMS

HOST = "0.0.0.0"
PORT = 9080


def getPredOut(stream:StreamData):
    res = stream.getResult()
    return res.getOutJPEG()

def getStreamOut(stream:StreamData):
    res = stream.getResult()
    return res.getInJPEG()

def getDirectOut(stream:StreamData):
    return stream.getJPEG()


# Main http server's stream handler
class StreamHandler(BaseHTTPRequestHandler):
    STREAM_MODES = {
        "/prediction/": getPredOut,
        "/direct/":     getDirectOut,
    }

    def invalidAddress(self):
        self.wfile.write(b"Yeah you typed in the wrong address, sorry.\r\n")
        self.wfile.write(f"Try something like: \"http://IP/prediction/X\", where X is the stream's ID\r\n".encode())
    
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

    def stream(self, stream_mode:int, stream_id:str):
        # Validation of address
        stream:StreamData = None
        try:
            stream = ALL_STREAMS[stream_id]
        except:
            self.invalidAddress()
            return
        # Address is valid after this point

        self.startStream()

        # Get stream objects
        stream = ALL_STREAMS[stream_id]
        jpeg_callback = StreamHandler.STREAM_MODES[stream_mode]

        while True:
            # Get jpeg from stream
            jpeg = jpeg_callback(stream)

            # Send stream data
            if jpeg is None:
                continue
            self.sendJPEG(jpeg)

    def do_GET(self):
        print(f"Client connecting at path {self.path}")

        # If streaming: Check mode
        for mode in self.STREAM_MODES.keys():
            stream_mode = self.path[0:len(mode)]
            if stream_mode == mode:
                break
            stream_mode = None

        try:
            if stream_mode is not None:
                stream_id = self.path[len(stream_mode):]
                self.stream(stream_mode, stream_id)
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