import multiprocessing as mp
import socket
import time

HOST = "localhost"
PORT = 1234
DURATION_S = 10
CHUNK_SIZE = 65536


def worker(label, outfile, start_delay):
    time.sleep(start_delay)
    total = 0
    try:
        with socket.create_connection((HOST, PORT), timeout=5) as sock:
            print(f"[{label}] TCP connect() succeeded")
            sock.settimeout(1.0)
            deadline = time.time() + DURATION_S
            with open(outfile, "wb") as f:
                while time.time() < deadline:
                    try:
                        data = sock.recv(CHUNK_SIZE)
                    except TimeoutError:
                        print("TIMEOUT")
                        continue
                    if not data:
                        print(f"[{label}] server closed the connection")
                        break
                    f.write(data)
                    total += len(data)
    except OSError as e:
        print(f"[{label}] connect() failed: {e}")
        return

    print(f"[{label}] received {total} bytes ({total / 1e6:.2f} MB) over {DURATION_S}s")


if __name__ == "__main__":
    p1 = mp.Process(target=worker, args=("client_A", "test_a.iq", 0.0))
    p2 = mp.Process(target=worker, args=("client_B", "test_b.iq", 0.3))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
