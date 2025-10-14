import os
import socket
import threading
import time
import webbrowser

from app import app

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = int(os.environ.get('PORT', 5002))


def find_open_port(start_port: int, host: str) -> int:
    """Return the first available port starting from start_port."""
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
        port += 1

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()
        return port


def wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait until the server accepts connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def open_browser_when_ready(url: str, host: str, port: int) -> None:
    """Open the default browser once the local server is reachable."""
    if wait_for_server(host, port):
        webbrowser.open_new_tab(url)
    else:
        webbrowser.open_new_tab(url)


def main() -> None:
    host = DEFAULT_HOST
    port = find_open_port(DEFAULT_PORT, host)
    url = f'http://{host}:{port}/'

    # Launch browser in a helper thread so the Flask reloader does not duplicate the tab.
    thread = threading.Thread(target=open_browser_when_ready, args=(url, host, port), daemon=True)
    thread.start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
