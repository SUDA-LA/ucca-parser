from ucca_parser import UCCA_Parser
from ucca.convert import passage2file, to_json
import os
from ucca import layer0, layer1
import json
import re
from collections import defaultdict


def node_label(node):
    return re.sub("[^(]*\((.*)\)", "\\1", node.attrib.get("label", ""))


def topological_layout(passage):
    visited = defaultdict(set)
    pos = {}
    terminals = passage.layer(layer0.LAYER_ID).all
    if terminals:
        implicit_offset = list(
            range(0, 1 + max(n.position for n in terminals)))
        leaves = sorted([n for layer in passage.layers for n in layer.all if not n.children],
                        key=lambda n: getattr(n, "position", None) or (n.fparent.end_position if n.fparent else -1))
        for node in leaves:  # draw leaves first to establish ordering
            if node.layer.ID == layer0.LAYER_ID:  # terminal
                x = node.position
                pos[node.ID] = (x + sum(implicit_offset[:x + 1]), 0)
            elif node.fparent:  # implicit
                implicit_offset[node.fparent.end_position] += 1
    else:
        implicit_offset = [0]
    remaining = [n for n in passage.layer(
        layer1.LAYER_ID).all if not n.parents]
    implicits = []
    while remaining:  # draw non-terminals
        node = remaining.pop()
        if node.ID in pos:  # done already
            continue
        children = [
            c for c in node.children if c.ID not in pos and c not in visited[node.ID]]
        if children:
            visited[node.ID].update(children)  # to avoid cycles
            remaining += [node] + children
            continue
        if node.children:
            xs, ys = zip(
                *(pos[c.ID] for c in node.children if not c.attrib.get("implicit")))
            pos[node.ID] = sum(xs) / len(xs), 1 + max(ys)  # done with children
        else:
            implicits.append(node)
    for node in implicits:
        fparent = node.fparent or passage.layer(layer1.LAYER_ID).heads[0]
        x = fparent.end_position
        x += sum(implicit_offset[:x + 1])
        _, y = pos.get(fparent.ID, (0, 0))
        pos[node.ID] = (x, y - 1)
    # stretch up to avoid over cluttering
    pos = {i: (x, y ** 1.01)for i, (x, y) in pos.items()}
    return pos


def analysis(passage):
    node_ids = False

    terminals = [{"id": n.ID, "label": n.text}
                 for n in passage.layer(layer0.LAYER_ID).all]
    implicits = [{"id": n.ID, "label": "IMPLICIT"} for n in passage.layer(layer1.LAYER_ID).all
                 if n.attrib.get("implicit")]
    nodes = [{"id": n.ID, "label": node_label(n) or (n.ID if node_ids else "")}
             for n in passage.layer(layer1.LAYER_ID).all if not n.attrib.get("implicit")]

    edges = [{"from": n.ID, "to": e.child.ID, "label": e.tag,
              "style": "dashed" if e.attrib.get("remote") else "solid"}
             for layer in passage.layers for n in layer.all for e in n]
    nodes = terminals + implicits + nodes
    pos = topological_layout(passage)
    for node in nodes:
        node["x"] = pos[node["id"]][0]
        node["y"] = pos[node["id"]][1]
    return nodes, edges

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
parser = UCCA_Parser.load("exp/char")
parser.eval()

words = ["After", "graduation", ",", "Mary", "moved", "to", "New", "York", "City"]

passage = parser.predict(words)

analysis(passage)
