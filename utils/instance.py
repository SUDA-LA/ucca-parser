import copy
from parser import UCCA2tree

from ucca.layer1 import FoundationalNode
from ucca.layer0 import Terminal


class Instance(object):
    def __init__(self, passage):
        self.passage = passage

        terminals = [
            (node.text, node.extra["pos"], node.extra["dep"], node.extra["ent_type"], node.extra["ent_iob"])
            for node in sorted(passage.layer("0").all, key=lambda x: x.position)
        ]
        words, pos, dep, entity, ent_iob = zip(*terminals)
        self.words = list(words)
        self.pos = list(pos)
        self.dep = list(dep)
        self.entity = list(entity)
        self.ent_iob = list(ent_iob)
        self.tree = self.generate_tree()

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

        all_span = []
        new_id = {}
        remote_edge = []
        if "1" in self.passage._layers:
            for node in self.passage.layer("1").all:
                for i in node._incoming:
                    if i.attrib.get("remote") == True:
                        remote_edge.append((i.child.ID, i.parent.ID, i.tag))
                        all_span.append(get_span(i.parent))
                        new_id[i.parent.ID] = len(new_id)

                        if i.child.ID not in new_id:
                            all_span.append(get_span(i.child))
                            new_id[i.child.ID] = len(new_id)

                if isinstance(node, FoundationalNode) and not node.attrib.get(
                    "implicit"
                ):
                    if node.ID not in new_id and not all(
                        isinstance(n, Terminal) for n in node.children
                    ):
                        all_span.append(get_span(node))
                        new_id[node.ID] = len(new_id)

        remote_edge = [(new_id[x], new_id[y], z) for x, y, z in remote_edge]
        return all_span, remote_edge
