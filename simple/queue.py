
from collections import deque

queue = deque()
queue.append(5)
queue.append(3)
queue.append(8)
queue.popleft() # pop from front of queue
queue.pop()  # pop from back

def level_order(root):
    if root == None:
        return
    q = deque()
    q.append(root)
    while len(q)>0:
        node = q.popleft()
        print(node.data, end=" ")
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)

def level_order_bylevel(root):
    if root is None:
        return
    q = deque()
    q.append(root)
    q.append(None)
    while len(q)>1:
        node = q.popleft()
        print(node.data, end=" ")
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
        q.append(None)
        if q[0] is None:
            print()
            q.popleft()

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

root = Node(10)
root.right = Node(30)
root.left = Node(20)
root.left.left = Node(40)

# level_order(root)
level_order_bylevel(root)