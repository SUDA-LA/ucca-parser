from ucca.core import edge_id_orderkey
from ucca.layer0 import Terminal
from ucca.layer1 import FoundationalNode, Layer1, NodeTags, PunctNode

from .trees import InternalTreebankNode, LeafTreebankNode


def remove_implicit(passage):
    for _, node in passage.nodes.items():
        if node.attrib.get("implicit"):
            node.destroy()


def remove_linkage(passage):
    for _, node in passage.nodes.items():
        if node.tag == "LKG":
            node.destroy()


def remove_remote(passage):
    for id, node in passage.nodes.items():
        to_remove = []
        for e in node.incoming:
            if e.attrib.get("remote"):
                to_remove.append(e)
        if len(to_remove) > 0:
            for e in to_remove:
                node._incoming.remove(e)
                e._parent._outgoing.remove(e)
            modify_edge = node.incoming[0]
            if hasattr(modify_edge, "categories"):
                modify_edge.categories[0]._tag += "-remote"
            else:
                modify_edge._tag += "-remote"


def remove_discontinue(passage):
    def terminal2id(terminals):
        return [int(t.ID.split(".")[1]) for t in terminals]

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
    
    def is_ancestor(ancestor, descendant):
        step = 0
        parent = descendant
        while True:
            if parent == ancestor:
                return step
            else:
                if len(parent.parents) == 0:
                    break
                parent = parent.parents[0]
                step += 1
        return -1

    def move(passage, dis_node): 
        def _down(terminal_node, dis_node, span):
            parent = terminal_node
            ext_label = None
            while True:
                assert len(parent.parents) == 1
                if parent.parents[0].discontiguous:
                    # ext_label = '-discontiguous'
                    ext_label = ''
                    break

                step = is_ancestor(parent.parents[0], dis_node)
                if step > -1:
                    ext_label = "-ancestor"
                    break
                parent = parent.parents[0]
                
            assert ext_label is not None
            modify_edge = parent.incoming[0]
            parent.parents[0]._outgoing.remove(modify_edge)
            parent.incoming[0]._parent = dis_node
            if ext_label == "-ancestor":
                if ext_label not in modify_edge.tag:
                    if step == 1:
                        if hasattr(modify_edge, "categories"):
                            modify_edge.categories[0]._tag += ext_label
                        else:
                            modify_edge._tag += ext_label
                else:
                    if hasattr(modify_edge, "categories"):
                        modify_edge.categories[0]._tag += modify_edge.categories[0]._tag[:1]
                    else:
                        modify_edge._tag = modify_edge._tag[:1]
            # if ext_label not in parent._incoming[0]._tag:
            #     parent._incoming[0].categories[0]._tag += ext_label
            dis_node._outgoing.append(modify_edge)
            dis_node._outgoing.sort(key=edge_id_orderkey)

        terminal = dis_node.get_terminals()
        discontinue_terminal = get_discontinue_terminal(passage, terminal)
        for t in discontinue_terminal:
            current_terminals = dis_node.get_terminals()
            if t not in current_terminals:
                assert len(t.parents) == 1
                _down(t, dis_node, terminal)

    while True:
        dis_nodes, dis_terminals = get_dis_nodes(passage)
        if len(dis_nodes) == 0:
            return
        for n in dis_nodes:
            move(passage, n)


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
    return to_treebank('ROOT', root[0])


def to_treebank(label, node):
    if len(node.outgoing) == 0:
        return LeafTreebankNode(node.extra["pos"], node.text)
    
    children = []
    for child in sorted(node.children, key=lambda x: x.get_terminals()[0].position):
        edge_label = child.incoming[0].tag
        children.append(to_treebank(edge_label, child))
    return InternalTreebankNode(label, children)


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
        for i in node.incoming:
            if '-ancestor' in i.tag:
                if len(i.parent.outgoing) > 1:
                        restore_down(node, i)
                if hasattr(i, "categories"):
                    i.categories[0]._tag = i.categories[0]._tag.strip("-ancestor")
                else:
                    i._tag = i._tag.strip("-ancestor")


def to_UCCA(passage, tree):
    passage = tree2passage(passage, tree)
    restore_discontinuity(passage)

    return passage
