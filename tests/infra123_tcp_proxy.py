"""Bounded PostgreSQL v3 TCP fault proxy, loopback ONLY, no packet logging.

Authentication bytes are forwarded opaquely and never retained. Tests disable
SSL/GSS solely for this disposable loopback protocol parser. Faults close real
sockets; they never manufacture a Python exception in application code.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

MAX_FRAME = 8 * 1024 * 1024


def is_commit(kind: bytes, payload: bytes) -> bool:
    if kind == b'Q':
        sql = payload.split(b'\0', 1)[0]
    elif kind == b'P':
        parts = payload.split(b'\0', 2)
        sql = parts[1] if len(parts) > 1 else b''
    else:
        return False
    return sql.strip().rstrip(b';').upper() == b'COMMIT'


class LocalPostgresProxy:
    def __init__(self, host: str, port: int):
        if host != '127.0.0.1' or not 0 < int(port) < 65536:
            raise ValueError('Test proxy accepts only a loopback PostgreSQL target')
        self.target = (host, int(port))
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.sessions = []
        self.pids = {}
        self.events = []

    def __enter__(self):
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(32)
        self.listener.settimeout(.2)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.listener.close()
        for session in list(self.sessions):
            session.abort()
        self.thread.join(2)
        for session in list(self.sessions):
            for thread in session.threads:
                thread.join(2)
        assert not self.thread.is_alive()
        assert not any(t.is_alive() for s in self.sessions for t in s.threads)

    def _accept(self):
        while not self.stop.is_set():
            try:
                client, address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            assert address[0] == '127.0.0.1'
            try:
                upstream = socket.create_connection(self.target, timeout=3)
            except OSError:
                client.close()
                continue
            session = _Session(self, client, upstream)
            with self.lock:
                self.sessions.append(session)
            session.start()

    def _session(self, pid):
        with self.lock:
            session = self.pids.get(pid)
        if session is None or session.closed.is_set():
            raise ValueError('Fault target is not an active owned proxy connection')
        return session

    def arm_commit(self, pid, mode):
        if mode not in {'partial_commit', 'drop_commit_reply'}:
            raise ValueError('Unsupported test fault')
        self._session(pid).mode = mode

    def cut(self, pid, reason):
        session = self._session(pid)
        self.record(session, reason, real_socket_close=True)
        session.abort()

    def record(self, session, mode, **fields):
        with self.lock:
            self.events.append({'fault': mode, 'backend_pid': session.pid,
                                'monotonic': time.monotonic(), **fields})


class _Session:
    def __init__(self, owner, client, upstream):
        self.owner, self.client, self.upstream = owner, client, upstream
        self.closed = threading.Event()
        self.abort_lock = threading.Lock()
        self.pid = None
        self.mode = None
        self.drop_reply = False
        self.threads = []
        for sock in (client, upstream):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(.2)

    def start(self):
        for target in (self.client_to_server, self.server_to_client):
            thread = threading.Thread(target=self._run, args=(target,), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _run(self, function):
        try:
            function()
        except (OSError, ValueError):
            pass
        finally:
            self.abort()

    def abort(self):
        with self.abort_lock:
            if self.closed.is_set():
                return
            self.closed.set()
            for sock in (self.client, self.upstream):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                    sock.close()
                except OSError:
                    pass

    def read(self, sock, length):
        parts = []
        remaining = length
        while remaining and not self.closed.is_set() and not self.owner.stop.is_set():
            try:
                data = sock.recv(remaining)
            except socket.timeout:
                continue
            if not data:
                return None
            parts.append(data)
            remaining -= len(data)
        return b''.join(parts) if not remaining else None

    def frame(self, sock):
        kind = self.read(sock, 1)
        if kind is None:
            return None
        header = self.read(sock, 4)
        if header is None:
            return None
        length = struct.unpack('!I', header)[0]
        if not 4 <= length <= MAX_FRAME:
            raise ValueError('Frame outside bounded test limit')
        payload = self.read(sock, length-4)
        return (kind, payload, kind+header+payload) if payload is not None else None

    def client_to_server(self):
        header = self.read(self.client, 4)
        if header is None:
            return
        length = struct.unpack('!I', header)[0]
        if not 8 <= length <= MAX_FRAME:
            raise ValueError('Invalid startup frame')
        startup = self.read(self.client, length-4)
        if startup is None or startup[:4] != struct.pack('!I', 196608):
            raise ValueError('Tests require plain PostgreSQL v3 on loopback')
        self.upstream.sendall(header+startup)
        while not self.closed.is_set():
            frame = self.frame(self.client)
            if frame is None:
                return
            kind, payload, raw = frame
            if self.mode and is_commit(kind, payload):
                if self.mode == 'partial_commit':
                    self.upstream.sendall(raw[:7])
                    self.owner.record(self, self.mode, real_socket_close=True,
                                      commit_bytes_forwarded=7, complete_frame_bytes=len(raw),
                                      complete_commit_forwarded=False)
                    return  # finally closes real sockets before completing frame
                self.drop_reply = True
                self.owner.record(self, self.mode, complete_commit_forwarded=True)
            self.upstream.sendall(raw)

    def server_to_client(self):
        while not self.closed.is_set():
            frame = self.frame(self.upstream)
            if frame is None:
                return
            kind, payload, raw = frame
            if kind == b'K' and len(payload) == 8:
                self.pid = struct.unpack('!I', payload[:4])[0]
                with self.owner.lock:
                    self.owner.pids[self.pid] = self
            if self.drop_reply:
                if kind == b'C' and payload == b'COMMIT\0':
                    self.owner.record(self, 'server_commit_reply_dropped', real_socket_close=True,
                                      server_command_complete_commit=True, reply_bytes_withheld=len(raw))
                    return  # close without forwarding confirmation to libpq
                continue
            self.client.sendall(raw)
