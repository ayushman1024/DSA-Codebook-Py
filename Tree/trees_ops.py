def tree_size(root):
    """ time O(N) and aux space O(h)"""
    if root is None:
        return 0

    return 1 + tree_size(root.left) + tree_size(root.right)

def max_in_tree(root):
    if root is None:
        return float('-inf')
    if root.left is None and root.right is None:
        return root.data
    return max( max_in_tree(root.left), max_in_tree(root.right) )

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

root = Node(10)
root.right = Node(30)
root.right.right = Node(60)
root.left = Node(20)
root.left.left = Node(40)
root.left.right = Node(50)

print(tree_size(root))
print(max_in_tree(root))