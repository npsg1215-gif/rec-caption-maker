"""녹화 + 자막 앱 실행기: 이 폴더를 http://localhost:8765 으로 열어 줍니다."""
import http.server
import os
import socket
import sys
import threading
import webbrowser

PORT = 8765
URL = f"http://localhost:{PORT}/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    # 윈도우 레지스트리 설정에 따라 .js 가 text/plain 으로 나가는 문제를 막습니다.
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def port_in_use():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


if __name__ == "__main__":
    open_browser = "--no-browser" not in sys.argv
    if port_in_use():
        # 이미 실행 중이면 브라우저만 엽니다.
        if open_browser:
            webbrowser.open(URL)
    else:
        with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as server:
            if open_browser:
                threading.Timer(0.7, lambda: webbrowser.open(URL)).start()
            print(f"앱이 브라우저에서 열렸어요: {URL}")
            print("이 창을 닫으면 앱이 꺼집니다.")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
