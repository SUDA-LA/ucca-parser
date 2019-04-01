import copy
from parser.convert import UCCA2tree

from ucca.layer1 import FoundationalNode
from ucca.layer0 import Terminal


class Instance(object):
    def __init__(self, passage):
        self.passage = passage

        terminals = [
            (node.text, node.extra["pos"])
            for node in sorted(passage.layer("0").all, key=lambda x: x.position)
        ]
        words, pos = zip(*terminals)
        self.words = list(words)
        self.pos = list(pos)
        self.tree = self.generate_tree()
        self.remote = self.gerenate_remote()

    @property
    def size(self):
        return len(self.words)

    def generate_tree(self):
        temp_passage = copy.deepcopy(self.passage)
        if "1" in self.passage._layers:
            return UCCA2tree(temp_passage)
        else:
            return None

    def gerenate_remote(self):
        def get_span(node):
            terminals = node.get_terminals()
            return (terminals[0].position - 1, terminals[-1].position)

        if "1" not in self.passage._layers:
            return [], ([], [], [])
        edges, spans = [], []
        nodes = [
            node
            for node in self.passage.layer("1").all
            if isinstance(node, FoundationalNode) and not node.attrib.get("implicit")
        ]
        ndict = {node: i for i, node in enumerate(nodes)}
        spans = [get_span(i) for i in nodes]

        remote_nodes = []
        for node in nodes:
            for i in node.incoming:
                if i.attrib.get("remote"):
                    remote_nodes.append(node)
                    break
        heads = [[ndict[n]] * len(nodes) for n in remote_nodes]
        deps = [list(range(len(nodes))) for _ in remote_nodes]
        labels = [["<NULL>"] * len(nodes) for _ in remote_nodes]
        for id, node in enumerate(remote_nodes):
            for i in node.incoming:
                if i.attrib.get("remote"):
                    labels[id][ndict[i.parent]] = i.tag

        return spans, (heads, deps, labels)
