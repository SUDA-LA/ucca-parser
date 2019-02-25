from ucca.convert import to_text, xml2passage
from ucca.core import edge_id_orderkey
from ucca.layer0 import Terminal
from ucca.layer1 import FoundationalNode, Layer1, NodeTags, PunctNode

from .trees import InternalTreebankNode, LeafTreebankNode


def remove_implicit(passage):
    for id, node in passage.nodes.items():
        if node.attrib.get("implicit") is True:
            incoming = node._incoming
            for p in node.parents:
                for e in incoming:
                    p._outgoing.remove(e)
            passage.layer("1")._all.remove(node)
            for e in incoming:
                passage.layer("1")._remove_edge(e)


def remove_linkage(passage):
    for id, node in passage.nodes.items():
        if node.tag == "LKG":
            outgoing = node._outgoing
            for p in node.children:
                for e in outgoing:
                    if e in p._incoming:
                        p._incoming.remove(e)
            passage.layer("1")._all.remove(node)
            for e in outgoing:
                passage.layer("1")._remove_edge(e)


def remove_remote(passage):
    for id, node in passage.nodes.items():
        to_remove = []
        for e in node._incoming:
            if e.attrib.get("remote") == True:
                to_remove.append(e)
        if len(to_remove) > 0:
            for e in to_remove:
                node._incoming.remove(e)
                e._parent._outgoing.remove(e)
            node._incoming[0]._tag += "".join(["-remote"])


def remove_discontinue(passage):
    def terminal2id(terminals):
        return [t.position for t in terminals]

    def get_discontinue_terminal(passage, terminal):
        id = terminal2id(terminal)
        left = id[0]
        right = id[-1]
        discontinue_terminal = []
        for i in range(left, right + 1):
            id = "0." + str(i)
            t = passage.nodes[id]
            if t not in terminal:
                discontinue_terminal.append(t)
        return discontinue_terminal

    def get_dis_nodes(passage):
        dis_terminals = []
        dis_nodes = []
        for node in passage.layer("1")._all:
            if node.discontiguous:
                dis_nodes.append(node)
                dis_terminals.append(node.get_terminals())
        return dis_nodes, dis_terminals

    def classify(dis_nodes, dis_terminals):
        with_overlap = []
        span = [
            set(range(int(t[0].ID.split(".")[1]), int(t[-1].ID.split(".")[1]) + 1))
            for t in dis_terminals
        ]
        for i in range(len(span) - 1):
            for j in range(i + 1, len(span)):
                re = span[i] & span[j]
                if len(re) > 0:
                    if dis_nodes[i] not in with_overlap:
                        with_overlap.append(dis_nodes[i])
                    if dis_nodes[j] not in with_overlap:
                        with_overlap.append(dis_nodes[j])
        without_overlap = [x for x in dis_nodes if x not in with_overlap]
        return without_overlap, with_overlap

    def move_down(passage, dis_node):
        def down(terminal_node, dis_node, span):
            parent = terminal_node
            while not set(parent.parents[0].get_terminals()) >= set(span):
                parent = parent.parents[0]
            modify_edge = parent._incoming[0]

            parent.parents[0]._outgoing.remove(modify_edge)
            parent._incoming[0]._parent = dis_node
            if "down" not in parent._incoming[0]._tag:
                parent._incoming[0]._tag += "-down"
            dis_node._outgoing.append(modify_edge)
            dis_node._outgoing.sort(key=edge_id_orderkey)

        terminal = dis_node.get_terminals()
        discontinue_terminal = get_discontinue_terminal(passage, terminal)
        for t in discontinue_terminal:
            current_terminals = dis_node.get_terminals()
            if t not in current_terminals:
                assert len(t.parents) == 1
                down(t, dis_node, terminal)

    def move_left(passage, left_dis_node, right_dis_node):
        assert left_dis_node.parents[0] == right_dis_node.parents[0]
        left_terminals, right_terminals = (
            left_dis_node.get_terminals(),
            right_dis_node.get_terminals(),
        )
        for n in get_discontinue_terminal(passage, left_terminals):
            if n not in left_dis_node.get_terminals():
                if n in right_terminals:
                    parent = n
                    while parent.parents[0] != right_dis_node:
                        parent = parent.parents[0]
                    modify_edge = parent._incoming[0]
                    moved_node = modify_edge.child

                    parent.parents[0]._outgoing.remove(modify_edge)
                    parent._incoming[0]._parent = left_dis_node
                    # if "left" not in parent._incoming[0]._tag:
                    #     parent._incoming[0]._tag += "-left"
                    left_dis_node._outgoing.append(modify_edge)
                    left_dis_node._outgoing.sort(key=edge_id_orderkey)

    while True:
        dis_nodes, dis_terminals = get_dis_nodes(passage)
        if len(dis_terminals) == 0:
            return
        without_overlap, with_overlap = classify(dis_nodes, dis_terminals)
        for n in without_overlap:
            move_down(passage, n)
        count = 0
        for i in range(len(with_overlap) - 1):
            x_t = with_overlap[i].get_terminals()
            x_span = range(x_t[0].position, x_t[-1].position + 1)
            for j in range(i + 1, len(with_overlap)):
                y_t = with_overlap[j].get_terminals()
                y_span = range(y_t[0].position, y_t[-1].position + 1)
                if set(x_span) >= set(y_span):
                    move_down(passage, with_overlap[j])
                    count += 1
                elif set(x_span) <= set(y_span):
                    move_down(passage, with_overlap[i])
                    count += 1
        if count == 0 and len(without_overlap) == 0:
            break

    without_overlap, with_overlap = classify(dis_nodes, dis_terminals)
    assert len(without_overlap) == 0
    without_overlap.sort(key=lambda x: x.position)
    move_left(passage, with_overlap[0], with_overlap[1])
    move_down(passage, with_overlap[0])


def UCCA2tree(passage):
    remove_implicit(passage)
    remove_linkage(passage)
    remove_remote(passage)
    try:
        remove_discontinue(passage)
    except Exception as e:
        print(passage.ID)

    root = [node for node in passage.layer("1")._all if len(node._incoming) == 0]
    assert len(root) == 1
    try:
        tree = to_tree(root[0])
    except Exception as e:
        print(passage.ID)
    assert len(tree) >= 2
    return to_treebank(tree)


def to_treebank(tree):
    if len(tree) == 1:
        pos = tree[0].split()[0]
        word = tree[0].split()[1]
        return LeafTreebankNode(pos, word)

    children = []
    for child in tree[1:]:
        children.append(to_treebank(child))
    return InternalTreebankNode(tree[0], children)


def to_tree(node):
    if len(node.children) == 0:
        return [node.extra["pos"] + " " + node.text]
    result = []

    if len(node._incoming) == 0:
        if node.tag == "LKG":
            result.append("ROOT-LKG")
        else:
            result.append("ROOT")
    else:
        result.append(node._incoming[0].tag)
    for child in sorted(node.children, key=lambda x: x.get_terminals()[0].position):
        result.append(to_tree(child))
    return result


def tree2passage(passage, tree):
    def add_node(passage, node, tree):
        global leaf_position
        if isinstance(tree, LeafTreebankNode):
            return
        for c in tree.children:
            if isinstance(c, LeafTreebankNode):
                node.add("Terminal", passage.layer("0").by_position(leaf_position))
                leaf_position += 1
            else:
                new_fnode = passage.layer("1").add_fnode(node, c.label)
                add_node(passage, new_fnode, c)

    def change_puncnode(passage):
        for t in passage.layer("0")._all:
            parent = t.parents[0]
            children = parent.children
            is_punc = sum(isinstance(c, Terminal) and c.punct for c in children)
            if is_punc == len(children):
                parent._tag = "PNCT"
                # true_ID = parent.ID
                # punc_node = PunctNode(
                #     root=passage,
                #     tag=NodeTags.Punctuation,
                #     ID=passage.layer("1").next_id(),
                # )
                # parent.parents[0].add(parent._incoming[0].tag, punc_node)
                # for c in children:
                #     punc_node.add("Terminal", c)

                # for i in parent._incoming:
                #     parent.parents[0]._outgoing.remove(i)
                # for i in parent._outgoing:
                #     i.child._incoming.remove(i)
                # passage.layer("1")._all.remove(parent)
                # punc_node._ID = true_ID

    passage = passage.copy(['0'])
    if "format" not in passage.extra:
        passage.extra["format"] = "ucca"

    layer = Layer1(passage, attrib=None)
    assert "0" in passage._layers
    if tree.label != "ROOT":
        print("warning: root label is not ROOT but %s" % tree.label)

    global leaf_position
    leaf_position = 1
    add_node(passage, layer._head_fnode, tree)
    change_puncnode(passage)
    return passage


def restore_discontinuity(passage):
    def restore_down(node, e):
        if len(node.parents) == 1 and len(node.parents[0].parents) == 1:
            # assert len(node.parents) == 1
            node.parents[0]._outgoing.remove(e)
            node.parents[0].parents[0]._outgoing.append(e)
            e._parent = node.parents[0].parents[0]
            node._outgoing.sort(key=edge_id_orderkey)

    def restore_left(node, e):
        assert len(node.parents) == 1
        parent = node.parents[0]
        children = list(
            sorted(parent.children, key=lambda x: x.get_terminals()[0].position)
        )
        from_index = children.index(node)
        if len(children) > 1:
            to_node = children[from_index - 1]
            node.parents[0]._outgoing.remove(e)
            e._parent = to_node
            to_node._outgoing.append(e)
            to_node._outgoing.sort(key=edge_id_orderkey)
       

    for node in passage.layer("1")._all:
        for i in node._incoming:
            if "down" in i.tag:
                if len(i.parent._outgoing) > 1:
                    restore_down(node, i)
                i._tag = i._tag.strip("-down")
            elif "left" in i.tag:
                restore_left(node, i)
                i._tag = i._tag.strip("-left")


def to_UCCA(passage, tree):
    passage = tree2passage(passage, tree)
    restore_discontinuity(passage)
    return passage
