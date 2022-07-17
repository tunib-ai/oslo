def dfs(node, bfs_dict=None):
    yield node
    if bfs_dict is not None:
        if node.depth in bfs_dict:
            bfs_dict[node.depth].append(node)
        else:
            bfs_dict[node.depth] = [node]

    for child in node.children:
        for c in dfs(child, bfs_dict):
            yield c


def bfs(node, bfs_dict=None):
    if bfs_dict is None:
        bfs_dict = {}
    if len(bfs_dict) == 0:
        list(dfs(node, bfs_dict))
    for nodes in bfs_dict.values():
        for node in nodes:
            yield node


def post_order_traverse(node):
    for child in node.children:
        yield from post_order_traverse(child)
    yield node


def is_iterable(data):
    try:
        iter(data)
    except TypeError:
        return False
    else:
        return True
