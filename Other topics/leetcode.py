

class MinStack:
    """https://leetcode.com/problems/min-stack/
    https://leetcode.com/problems/min-stack/solutions/4589441/o-n-space-complexity-solution-better-than-o-2n/"""

    def __init__(self):
        self.min_now = float("inf")
        self.stack = []

    def push(self, val: int) -> None:
        if len(self.stack) == 0:
            self.stack.append(val)
            self.min_now = val
        elif val < self.min_now:
            self.stack.append(2*val - self.min_now)
            self.min_now = val
        else:
            self.stack.append(val)

    def pop(self) -> None:
        if len(self.stack) == 0:
            return
        val = self.stack.pop()
        if val < self.min_now:
            self.min_now = 2*self.min_now - val

    def top(self) -> int:
        val = self.stack[-1]
        if val < self.min_now:
            return self.min_now
        return val

    def getMin(self) -> int:
        return self.min_now

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def treeQueries(self, root, queries):
        """ https://leetcode.com/problems/height-of-binary-tree-after-subtree-re
        moval-queries/
        Problem name: Height of binary tree after subtree removal queries
        Given a binary tree, we will remove the nodes of the tree that have a value equal to one of the nodes in the queries list.
        Return the height of the remaining tree after the removals.

        """
        hmap, lh, rh = dict(), dict(), dict() # height map, left height, right height

        def get_height(node):
            if node is None:
                return 0
            l = get_height(node.left)
            r = get_height(node.right)

            # Also populate height of left and right subtrees
            lh[node.val] = l
            rh[node.val] = r
            return 1 + max(l,r)
        
        def preorder(node, depth, height_g): # height_g is the height of the node after removal of node
            if node is None:
                return

            hmap[node.val] = height_g # height of the node after removal, computer in parent node
            
            preorder(node.left, depth+1, max(height_g, depth + rh[node.val]))
            preorder(node.right, depth+1, max(height_g, depth + lh[node.val]))
        
        get_height(root) # populate lh and rh
        preorder(root.left, 1, rh[root.val])
        preorder(root.right, 1, lh[root.val])

        output = []
        for q in queries:
            if q in hmap:
                output.append( hmap[q] )
        return output