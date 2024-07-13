# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque
from typing import List, Optional
class Solution:
    """https://leetcode.com/problems/find-leaves-of-binary-tree/submissions/1150983047/
    Below I have solved this using topological sort but it can also be solved by another approach
        -> We can calculate max height of each node from leaf node. height[node] = 1 + max(left_height, right_height)
        -> we can club together all the nodes height wise in ascedning order to generate result.
    """
    def rev_adj(self, root):
        if root is None:
            return
        if root not in self.adj:
            self.adj[root] = []
        if root.left:
            if root.left not in self.adj:
                self.adj[root.left] = []
            self.adj[root.left].append(root)
            self.rev_adj(root.left)
        if root.right:
            if root.right not in self.adj:
                self.adj[root.right] = []
            self.adj[root.right].append(root)
            self.rev_adj(root.right)

    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []

        self.adj = dict()
        self.rev_adj(root)
        nodes = self.adj.keys()
        indeg = dict()
        
        for node in nodes:
            indeg[node] = 0

        for node in self.adj:
            for tonode in self.adj[node]:
                indeg[tonode] += 1

        q = deque()
        for node in indeg:
            if indeg[node] == 0:
                q.append(node)

        while len(q):
            loopsize = len(q)
            order = []
            for _ in range(loopsize):
                node = q.popleft()
                order.append(node.val)
                for adj in self.adj[node]:
                    indeg[adj] -= 1
                    if indeg[adj] == 0:
                        q.append(adj)
            res.append(order)
        return res