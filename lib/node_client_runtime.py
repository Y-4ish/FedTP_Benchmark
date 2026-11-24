"""Run lockstep client trainers as coroutines inside four FATE participants."""

from collections import defaultdict
import pickle
from types import SimpleNamespace

from lib.node_client_layout import assign_logical_clients, dataset_node_count


MODELS = {"FedTSE", "TDLR", "SEDLR", "TDLR_SEDLR", "CNFGNN", "FedGODE",
          "FedOSTC", "REFOL", "FedSTG", "SFL_RNN", "FUELS"}
_CHUNK_BYTES = 512 * 1024


def send_packet(endpoint, tag, packet):
    data = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
    chunks = range(0, len(data), _CHUNK_BYTES)
    endpoint.put(tag, len(chunks))
    for index, start in enumerate(chunks):
        endpoint.put(f"{tag}_{index}", data[start:start + _CHUNK_BYTES])


def receive_packet(endpoint, tag):
    count = endpoint.get(tag)
    return pickle.loads(b"".join(endpoint.get(f"{tag}_{index}") for index in range(count)))


class Endpoint:
    def __init__(self, get, put):
        self.get = get
        self.put = put


class Arbiter:
    def __init__(self, physical, assignments):
        self.physical = physical
        self.assignments = assignments
        self.count = sum(map(len, assignments))
        self.rank = self.count
        self.is_on_arbiter = True
        self.is_on_guest = self.is_on_host = False
        self.guest = Endpoint(lambda tag: self.receive(tag, [0])[0],
                              lambda tag, value: self.send(tag, {0: value}))
        self.hosts = Endpoint(lambda tag: self.receive(tag, list(range(1, self.count))),
                              self.send_hosts)
        self.uploads = defaultdict(dict)
        self.downloads = defaultdict(dict)
        self.pending = None
        self.sequence = 0

    def collect(self):
        tag = f"node_up_{self.sequence}"
        packets = [receive_packet(party, tag)
                   for party in (self.physical.guest, *self.physical.hosts)]
        self.pending = []
        for nodes, packet in zip(self.assignments, packets):
            if set(packet) != set(nodes):
                raise RuntimeError("Worker returned incorrect logical client IDs")
            requests = {}
            for node, (uploads, request) in packet.items():
                for key, value in uploads.items():
                    self.uploads[key][node] = value
                requests[node] = request
            self.pending.append(requests)
        if len(self.pending) != len(self.assignments):
            raise RuntimeError("Missing worker packet")

    def flush(self):
        if self.pending is None:
            return
        if not all(key is None or node in self.downloads.get(key, {})
                   for requests in self.pending for node, key in requests.items()):
            return
        replies = [
            {node: self.downloads[key].pop(node)
             for node, key in requests.items() if key is not None}
            for requests in self.pending
        ]
        for key in list(self.downloads):
            if not self.downloads[key]:
                del self.downloads[key]
        if any(replies):
            tag = f"node_down_{self.sequence}"
            send_packet(self.physical.guest, tag, replies[0])
            for host, reply in zip(self.physical.hosts, replies[1:]):
                send_packet(host, tag, reply)
        self.pending = None
        self.sequence += 1

    def receive(self, tag, nodes):
        while not all(node in self.uploads.get(tag, {}) for node in nodes):
            self.flush()
            if self.pending is not None:
                raise RuntimeError(f"Clients are waiting for a reply before uploading {tag}")
            self.collect()
        bucket = self.uploads.get(tag, {})
        if not all(node in bucket for node in nodes):
            raise RuntimeError(f"Missing logical client uploads for {tag}")
        values = [bucket.pop(node) for node in nodes]
        if not bucket:
            del self.uploads[tag]
        return values

    def send(self, tag, values):
        self.downloads[tag].update(values)
        self.flush()

    def send_hosts(self, tag, values):
        if not isinstance(values, list) or len(values) != self.count - 1:
            values = [values] * (self.count - 1)
        self.send(tag, dict(enumerate(values, start=1)))

    def finish(self):
        # A trainer may publish a final checkpoint then return before workers
        # have reached the corresponding receive.
        while self.downloads:
            if self.pending is None:
                self.collect()
            self.flush()
            if self.pending is not None:
                raise RuntimeError("Unmatched final logical client messages")


class NodeEarlyStopper:
    def __init__(self, ctx, patience=50, min_delta=1e-4):
        self.ctx = ctx
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.stale = 0

    def step(self, epoch, loss=None, return_best=False):
        tag = f"node_val_{epoch}"
        decision = f"node_stop_{epoch}"
        if not self.ctx.is_on_arbiter:
            self.ctx.arbiter.put(tag, float(loss))
            return self.ctx.arbiter.get(decision)
        losses = [self.ctx.guest.get(tag), *self.ctx.hosts.get(tag)]
        mean_loss = sum(losses) / len(losses)
        is_best = mean_loss < self.best - self.min_delta
        if is_best:
            self.best, self.stale = mean_loss, 0
        else:
            self.stale += 1
        stop = self.stale >= self.patience
        payload = (is_best, stop) if return_best else stop
        self.ctx.guest.put(decision, payload)
        self.ctx.hosts.put(decision, payload)
        print(f"[Node Clients] epoch={epoch + 1} val_mae={mean_loss:.6f} "
              f"patience={self.stale}/{self.patience}", flush=True)
        return payload


def run_worker(physical, nodes, run_client, write_result, flush_uploads=False):
    from greenlet import getcurrent, greenlet
    import torch

    parent = getcurrent()
    pending = {node: {} for node in nodes}
    grad_state = {node: torch.is_grad_enabled() for node in nodes}
    parent_grad = torch.is_grad_enabled()

    def start(node):
        def suspend(request):
            grad_state[node] = torch.is_grad_enabled()
            torch.set_grad_enabled(parent_grad)
            return parent.switch(request)

        def put(tag, value):
            if tag in pending[node]:
                raise RuntimeError(f"Duplicate upload {tag} from node {node}")
            pending[node][tag] = value
            if flush_uploads:
                suspend(None)

        ctx = SimpleNamespace(
            rank=node, is_on_arbiter=False, is_on_guest=node == 0,
            is_on_host=node != 0, arbiter=Endpoint(suspend, put),
        )
        return run_client(ctx)

    clients = {node: greenlet(start) for node in nodes}
    replies = {node: node for node in nodes}
    sequence = 0
    while True:
        packet = {}
        for node, client in clients.items():
            torch.set_grad_enabled(grad_state[node])
            try:
                request = client.switch(replies.get(node))
            finally:
                torch.set_grad_enabled(parent_grad)
            if client.dead:
                write_result(node, request)
                request = None
            packet[node] = (pending[node], request)
            pending[node] = {}
        waiting = sum(request is not None for _, request in packet.values())
        if waiting not in (0, len(nodes)):
            raise RuntimeError("Logical clients exited at different communication stages")
        if waiting or any(uploads for uploads, _ in packet.values()):
            send_packet(physical.arbiter, f"node_up_{sequence}", packet)
        if all(client.dead for client in clients.values()):
            return
        if not waiting:
            replies = {}
            sequence += 1
            continue
        replies = receive_packet(physical.arbiter, f"node_down_{sequence}")
        if set(replies) != set(nodes):
            raise RuntimeError("Arbiter returned incorrect logical client IDs")
        sequence += 1


def initialize(args, root):
    if args.model not in MODELS:
        raise ValueError(f"Node client mode is not implemented for {args.model}")
    if args.num_clients != 4 or args.protection != "plain" or args.dp_noise:
        raise ValueError("Node client mode requires four FATE workers and plain protection")
    if getattr(args, "privacy_trace_only", False):
        raise ValueError("Node client mode does not support privacy traces")
    from greenlet import greenlet  # Check the dependency before launching FATE.

    count = dataset_node_count(root, args.dataset_name)
    return assign_logical_clients(count, args.num_clients)
