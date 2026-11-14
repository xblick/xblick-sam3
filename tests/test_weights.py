"""The node's weights (weights.py) and its --download-only and --no-download, without a GPU, the real weights or the
internet: the files, their checksums and sources, and the node's start up to the end of its weights step.

The download itself is the core's xblick.weights, which tests it in its own tests/test_weights.py. Downloads here go
to a local HTTP server: URL files directly, Hugging Face files through HF_ENDPOINT. Where network use must not
happen, any use fails the test. The part from "the same in every node's test_weights.py" on is the same in every
node."""

import contextlib
import hashlib
import http.server
import io
import os
import signal
import socket
import threading
from pathlib import Path

import pytest
from xblick import weights as xblick_weights
from xblick.session import open_session

from xblick_sam3 import node, weights

TERMS_LINE = "SAM 3 is the work of Meta"  # the first line of weights.TERMS
NETWORK = True  # the weights come from the network

WITHOUT_SHA256 = ()  # files whose source states no SHA-256


def start_options(tmp_path) -> list[str]:
    """A start's required options."""
    return ["--camera", "test/camera"]


def default_weights():
    return weights.needed()


@pytest.fixture
def stand_in(home, server, monkeypatch):
    """SAM 3.1's checkpoint as a stand-in on the local server, through HF_ENDPOINT: {path: content}."""
    monkeypatch.setenv("HF_ENDPOINT", server.url)
    monkeypatch.setattr(weights, "CHECKPOINT_SHA256", DATA_SHA)
    monkeypatch.setattr(weights, "CHECKPOINT_SIZE", len(DATA))
    server.files["/sam3.1_multiplex.pt"] = DATA
    return {home / "sam3.1_multiplex.pt": DATA}


def test_the_file_is_the_gated_sam31_checkpoint_with_its_sha256():
    [w] = weights.needed()
    assert w.path == weights.WEIGHTS / "sam3.1_multiplex.pt" == weights.CHECKPOINT
    assert weights.WEIGHTS == weights.REPO / "checkpoints"
    commit = "daa63191845a41281374e725f4c9e51c7a824460"  # main's commit on 2026-11-14, pinned
    assert w.source == weights.HuggingFace("facebook/sam3.1", commit, "sam3.1_multiplex.pt", gated=True)
    assert w.source.url().endswith(f"/facebook/sam3.1/resolve/{commit}/sam3.1_multiplex.pt")
    assert w.size == 3_502_755_717
    assert w.sha256 == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6" and w.size > 3.5e9
    assert weights.needed("/some/sam3.pt") == []  # a file of one's own: nothing to download


def test_without_access_the_node_says_how_to_get_it(home, server, stand_in, capsys):
    server.status = 403  # Hugging Face's answer to an account without access to a gated repository
    assert node.main(["--download-only"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "https://huggingface.co/facebook/sam3.1" in err[0]
    assert ".venv/bin/hf auth login" in err[0] and "HF_TOKEN" in err[0]
    assert server.gets == [("/facebook/sam3.1/resolve/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt",
                             "Bearer secret")]


# ------------------------------------------------------------------------ the same in every node's test_weights.py

DATA = os.urandom(3 * 1024 * 1024 + 17)  # a stand-in weight file, larger than a few read chunks
DATA_SHA = hashlib.sha256(DATA).hexdigest()


class Server:
    """A local HTTP server of files: path -> bytes. ``/resolve/`` paths redirect to ``/cdn/`` (as Hugging Face's do);
    ``status``: every GET answered with it. Records each GET and its Authorization header."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.gets: list[tuple[str, str | None]] = []
        self.status: int | None = None
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.gets.append((self.path, self.headers.get("Authorization")))
                if owner.status:
                    self.send_error(owner.status)
                    return
                if "/resolve/" in self.path:
                    self.send_response(302)
                    self.send_header("Location", "/cdn/" + self.path.rsplit("/", 1)[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = owner.files.get(self.path.removeprefix("/cdn"))
                if data is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(OSError):
                    self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture
def no_network(monkeypatch):
    """Any network use of this process fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"network use: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def use_store(monkeypatch, required=()):
    """The node's STORE as it is, with other required paths (code from the setup)."""
    store = weights.STORE
    monkeypatch.setattr(weights, "STORE", xblick_weights.Store(store.prog, store.root, store.terms, required,
                                                               repo=store.repo))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """weights.WEIGHTS and the STORE's root (the lock goes there) in tmp_path; no code from the setup required
    (stand_in may); a token that is no login of this PC."""
    root = tmp_path / "weights"
    monkeypatch.setattr(weights, "WEIGHTS", root)
    monkeypatch.setattr(weights, "STORE", xblick_weights.Store(weights.PROG, root, weights.TERMS, (),
                                                               repo=weights.REPO))
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return root


def place(files: dict[Path, bytes]):
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_the_store_is_the_nodes():
    store = weights.STORE
    assert (store.prog, store.root, store.terms, store.required, store.repo) == (
        weights.PROG, weights.WEIGHTS, weights.TERMS, tuple(weights.REQUIRED), weights.REPO)
    assert store.prog == node.main.__module__.split(".")[0].replace("_", "-")


def test_every_file_of_the_node_has_its_exact_size_and_sha256():
    for w in default_weights():
        assert isinstance(w, xblick_weights.Weight) and w.size > 0, w.path.name
        assert (w.sha256 is None) == (w.path.name in WITHOUT_SHA256), w.path.name
        assert w.sha256 is None or len(w.sha256) == 64


def test_all_present_touches_neither_network_nor_disk(home, stand_in, no_network):
    place(stand_in)
    before = sorted(home.rglob("*"))
    out = io.StringIO()
    assert weights.STORE.ensure(default_weights(), out=out) == [] and out.getvalue() == ""
    assert sorted(home.rglob("*")) == before  # no lock file either


def test_download_only_with_all_present_prints_one_line(home, stand_in, no_network, capsys):
    place(stand_in)
    assert node.main(["--download-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and "nothing to download" in lines[0]


def test_download_only_downloads_with_the_terms_and_exits_0(home, stand_in, capsys):
    assert node.main(["--download-only"]) == 0
    assert all(path.read_bytes() == data for path, data in stand_in.items())
    out = capsys.readouterr().out
    assert TERMS_LINE in out and f"weights complete: {len(stand_in)} downloaded" in out.splitlines()[-1]
    assert not list(home.rglob("*.part"))


def test_download_only_needs_no_start_options_and_excludes_no_download(capsys):
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--no-download"])
    assert exit_.value.code == 2 and "not allowed with" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_:
        node.main(["--no-download"])  # a start needs its required options
    assert exit_.value.code == 2 and "required" in capsys.readouterr().err


class Started(Exception):
    """The node got past its weights: what follows (the model, the GPU) is not this test's."""


@pytest.fixture
def started(monkeypatch, tmp_path):
    """The STORE's ensure raising Started once it returned (the node's start stops there, before anything touches
    the GPU: the tests need none); a Zenoh session the node connects to; the node's parameters kept."""
    from xblick import node as core_node

    ensure = weights.STORE.ensure

    def ensure_then_stop(*args, **kwargs):
        ensure(*args, **kwargs)
        raise Started

    monkeypatch.setattr(weights.STORE, "ensure", ensure_then_stop)
    kept = []

    class Kept(core_node.Parameters):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kept.append(self)

    monkeypatch.setattr(core_node, "Parameters", Kept)
    sessions = []
    opened = node.open_from_args

    def keep_session(args):
        sessions.append(opened(args))
        return sessions[-1]

    monkeypatch.setattr(node, "open_from_args", keep_session)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    sigterm = signal.getsignal(signal.SIGTERM)  # node.start sets its own
    yield [*start_options(tmp_path), "--connect", f"127.0.0.1:{port}"], kept
    for closable in (*kept, *sessions):  # a start that got past the weights left them open (Started)
        with contextlib.suppress(Exception):  # closed already by a node that ended
            closable.close()
    signal.signal(signal.SIGTERM, sigterm)
    session.close()


def test_no_download_ends_at_once_naming_the_missing_files(home, server, stand_in, started, capsys):
    start, kept = started
    assert node.main([*start, "--no-download"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and all(path.name in err[0] for path in stand_in) and "--download-only" in err[0]
    assert server.gets == [] and not any(path.exists() for path in stand_in)
    # a read-only parameter, as every start option; --download-only is a mode of the command line, no parameter
    assert kept[0].descriptor("no_download").read_only and kept[0]["no_download"] is True
    assert "download_only" not in kept[0].names()


def test_no_download_with_all_present_starts_as_usual(home, stand_in, started, no_network):
    start, _ = started
    place(stand_in)
    with pytest.raises(Started):
        node.main([*start, "--no-download"])


def test_a_start_downloads_what_is_missing_first(home, stand_in, started):
    start, _ = started
    with pytest.raises(Started):
        node.main(start)
    assert all(path.read_bytes() == data for path, data in stand_in.items())
