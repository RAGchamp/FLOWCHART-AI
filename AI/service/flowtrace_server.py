"""FlowTrace local service for the Flow Chart App's "Trace it" button.

    python flowtrace_server.py            (or double-click AI\\start-flowtrace.bat)

Listens on http://127.0.0.1:8765 only (this computer). Endpoints:
    GET  /health    → {ok, version, model, ocr}
    POST /corners   {image}                       → {corners, conf, width, height}
    POST /trace     {image, corners, detect, detail, debug} → diagram plan (INFO/FLOW-CHART-AI-PLAN.md §4)
"""
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels      # noqa: E402
import pipeline    # noqa: E402
import segment     # noqa: E402

HOST, PORT = '127.0.0.1', int(os.environ.get('FLOWTRACE_PORT', 8765))
MAX_BODY = 40 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = 'FlowTrace/' + pipeline.VERSION

    def cors(self):
        # the app is opened from a file:// page or its web address; both may call this local service
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Private-Network', 'true')

    def reply(self, code, obj):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        if self.path.split('?')[0] == '/health':
            self.reply(200, {'ok': True, 'version': pipeline.VERSION,
                             'model': 'model' if os.path.exists(segment.MODEL_PATH) else 'classic',
                             'ocr': labels.tesseract_ok()})
        else:
            self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        path = self.path.split('?')[0]
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > MAX_BODY:
            return self.reply(413, {'error': 'The picture is missing or too large (over 40 MB).'})
        try:
            req = json.loads(self.rfile.read(n))
        except ValueError:
            return self.reply(400, {'error': 'The request is not valid JSON.'})
        if not isinstance(req, dict) or not req.get('image'):
            return self.reply(400, {'error': 'No picture was sent.'})
        t = time.time()
        try:
            if path == '/trace':
                out = pipeline.trace(req)
            elif path == '/corners':
                out = pipeline.corners(req)
            else:
                return self.reply(404, {'error': 'Not found'})
        except ValueError as e:
            return self.reply(400, {'error': str(e)})
        except Exception as e:
            traceback.print_exc()
            return self.reply(500, {'error': 'Tracing failed: ' + str(e)})
        print(f'{path}: {time.time() - t:.1f} s')
        self.reply(200, out)

    def log_message(self, fmt, *args):
        pass


def main():
    # on Windows, address reuse would let a second copy share the port silently (and answer with old
    # code); without it, a second start fails with a clear message instead
    ThreadingHTTPServer.allow_reuse_address = False
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    ocr = labels.tesseract_ok()
    print(f'FlowTrace {pipeline.VERSION} on http://{HOST}:{PORT}')
    print('  segmentation:', 'trained model' if os.path.exists(segment.MODEL_PATH) else 'classic (no trained model yet)')
    print('  text reading:', f'Tesseract {ocr}' if ocr else 'NOT FOUND - install Tesseract to read labels')
    print('Leave this window open while you use Trace it in the Flow Chart App. Ctrl+C stops it.')
    server.serve_forever()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f'Could not start on port {PORT}: {e}')
        print('FlowTrace is probably already running in another window. Close that window first, then start this again.')
        sys.exit(1)
