from collections import deque

def level_order(root):
    """ init queue with root, pop queue to print and enq its left/right child.
       repeat until queue is empty"""
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

def level_order_by_level_1(root):
    """
    time O(n)
    aux space  theta(width)

    We use Null value to identify end of each level
    """
    if root is None:
        return
    q = deque()
    q.append(root)
    q.append(None) # end of level
    while len(q)>1:        # notice looping till q size>1
        node = q.popleft()
        print(node.data, end=" ")
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)

        if q[0] is None:
            q.popleft()
            q.append(None)
            print(" : ", end="")

def level_order_by_level_2(root):
    if root == None:
        return
    q = deque()
    q.append(root)
    while len(q)>0:
        loopsize = len(q)
        for _ in range(loopsize):
            node = q.popleft()
            print(node.data, end=" ")
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        print(" : ", end="")

def preorder_iterative(root):
    output = []
    stack = []
    stack.append(root)
    while len(stack):
        node = stack.pop()
        output.append(node.data)
        if node.right:    # notice right is added first in stack  to ensure  N L R order
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return output

def postorder_iterative(root):
    """ We need two stack. Every node poped from stack1 goes to stack2. reversed stack2 will be output.
    1. Init stack1 with root, and run loop with below logic till stack1 is empty
    2. Pop top of stack1 and move it to stack2. And insert its Right and Left nodes back to stack1 is not null.
    """
    stk1, stk2 = [], []
    node = root
    stk1.append(node)
    
    while stk1:
        nodei = stk1.pop()
        if nodei.left:
            stk1.append(nodei.left)
        if nodei.right:
            stk1.append(nodei.right)
        stk2.append(nodei.data)

    return stk2[::-1]

def inorder_iterative(root):

    """ https://leetcode.com/problems/binary-tree-inorder-traversal/submissions/1142595148/
        Solution: https://leetcode.com/problems/binary-tree-inorder-traversal/solutions/4542238/iterative-2-stack-solution-solution-python-3/

    create "node" variable initialised to root. Traverse whole tree by first keep going left till null 
    and then start poping stack and initialising node to right of poped node"""

    stack, output = [], []
    node = root
    while stack or node is not None:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            output.append(node.val)
            node = node.right
    return output

def check_balance(root):
    """ By finding left and right height at each node. 
    We return -1 as height when any of the subtree is unbalanced to indicate complete tree can be decalred unbalanced.
    Else return 1 + max(left_height, right_height)"""

    if root is None:
        return 0
    left_height,right_height = 0,0
    if root.left:
        left_height = check_balance(root.left)
    if root.right:
        right_height = check_balance(root.right)
    if right_height < 0 or left_height < 0:
        return -1
    diff = abs(left_height - right_height)
    if diff > 1 :
        return -1
    else:
        return 1 + max(left_height, right_height)


def widthOfBinaryTree(root) -> int:
    """  https://leetcode.com/problems/maximum-width-of-binary-tree/ 
         https://takeuforward.org/data-structure/maximum-width-of-a-binary-tree/ """
    q = deque()
    q.append([root,0])
    ans = -1
    while len(q)>0:
        loop = len(q)
        min_,max_ = float("inf"), float("-inf")
        for _ in range(loop):
            node, p = q.popleft()
            min_, max_ = min(min_,p), max(max_,p)
            if node.left:
                q.append([node.left, 2*p + 1])
            if node.right:
                q.append([node.right, 2*p + 2])
            
        ans = max(ans, max_-min_+1)
    return ans

class Problems:
    """https://takeuforward.org/data-structure/maximum-width-of-a-binary-tree/"""
    def __init__(self):
        self.max_w = 0

    def max_width(self, root):
        if root is None:
            return 0
        l_width = self.max_width(root.left)
        r_width = self.max_width(root.right)

        self.max_w = max(self.max_w, r_width + l_width) # update global variable

        return 1 + max(l_width,r_width)

def get_leafs(root, arr):
    if root is None:
        return
    if root.left is None and root.right is None:
        arr.append(root.data)
    get_leafs(root.left, arr)
    get_leafs(root.right, arr)

def boundary_trav(root):
    """
    Anti-clockwise boundary traversal
    1. left boundary +  2. leaf nodes  + 3. right boundary in reverse
        note: do not include leaf node in left/right boundary & include root node only once in left boundary"""
    # left boundary ( excluding leaf at left boundary )
    output = []
    node = root

    while(True):
        if node.left:
            output.append(node.data)
            node = node.left
        elif node.right:
            node = node.right
        else:
            break
    # all leaf nodes ( leaf to right )
    leafs = []
    get_leafs(root,leafs)
    output += leafs

    # Right boundary (excluding root node and leaf at right boundary)
    right_boundary = []
    node = root.right
    while(True):
        if node.right:
            right_boundary.append(node.data)
            node = node.right
        elif node.left:
            node = node.left
        else:
            break
    output += right_boundary[::-1] # reverse right boundary
    return output

def right_boundary( root ):
    res = []
    node = root
    if node is None:
        return []
    while True:
        res.append(node.val)
        if node.right:
            node = node.right
        elif node.left:
            node = node.left
        else:
            break
    return res

def boundaryOfBinaryTree( root ):
    """https://leetcode.com/problems/boundary-of-binary-tree/
    https://www.youtube.com/watch?v=0ca1nvR0be4&list=PLkjdNRgDmcc0Pom5erUBU4ZayeU9AyRRu&index=20"""
    def get_leftboundary(node):
        if node:
            if node.left:
                out.append(node.val)
                get_leftboundary(node.left)
            elif node.right:
                out.append(node.val)
                get_leftboundary(node.right)

    def get_leafs( node):
        if node is None:
            return
        if node.left is None and node.right is None and node is not root: # if only one node in tree, root shouldn't counted as leaf
            out.append(node.val)

        get_leafs(node.left)
        get_leafs(node.right)
    
    def get_rightboundary(node):
        if node:
            if node.right:
                get_rightboundary(node.right)
                out.append(node.val)
            elif node.left:
                get_rightboundary(node.left)
                out.append(node.val)

    out = [root.val]

    get_leftboundary(root.left)
    get_leafs(root)
    get_rightboundary(root.right)

    return out

def verticalTraversal( root ):
    """https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/
        https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/solutions/4546730/modified-level-order-traversal-approach-python-3/
    
    """
    verticals = dict()
    q = deque()
    q.append([root,0])
    while q:
        loop = len(q)
        level = []
        for _ in range(loop):
            node, ver = q.popleft()
            level.append([node.val,ver])
            if node.left:
                q.append([node.left, ver-1])
            if node.right:
                q.append([node.right, ver+1])
        
        # sort by values
        level = sorted(level)
        for val,lev in level:
            if lev not in verticals:
                verticals[lev] = []
            verticals[lev].append(val)
    levels = sorted(verticals)
    res = []
    for lev in levels:
        res.append(verticals[lev])
    return res

def vertical_order_traversal(root):
    level_map = dict()
    q = deque()
    q.append([root,0])
    min_key = float('inf')
    max_key = float('-inf')
    while(len(q)):
        data = q.popleft()
        h = data[1]
        if h not in level_map:
            level_map[h] = []
        level_map[h].append(data[0].data)
        if data[0].left:
            q.append([data[0].left,h-1])
            min_key = min(min_key,h-1)
        if data[0].right:
            q.append([data[0].right,h+1])
            max_key = max(max_key,h+1)
    for key in range(min_key,max_key+1):
        if key in level_map:
            print(level_map[key], end=" ")

def topView(root):
    """ https://www.geeksforgeeks.org/problems/top-view-of-binary-tree/1
    Modified vertical order traversal to skip adding new element in each level. Recursive solution (going left and rightwill not work )"""
    
    # code here
    verticals = dict()
    q = deque()
    q.append([root,0])
    
    while len(q)>0:
        size = len(q)
        for _ in range(size):
            node, ver = q.popleft()
            if ver not in verticals:
                verticals[ver] = node.data
            if node.left:
                q.append([node.left, ver-1])
            if node.right:
                q.append([node.right, ver+1])
            
    vers = sorted(verticals)
    res = []
    for ver in vers:
        res.append(verticals[ver])
    return res

# def OLDtopview(root):
#     level_map = dict()
#     q = deque()
#     q.append([root,0])
#     min_key = float('inf')
#     max_key = float('-inf')
#     while(len(q)):
#         data = q.popleft()
#         h = data[1]
#         if h not in level_map:
#             level_map[h] = None
#             level_map[h] = data[0].data  # this will be skipped for existing vertical
#         if data[0].left:
#             q.append([data[0].left,h-1])
#             min_key = min(min_key,h-1)
#         if data[0].right:
#             q.append([data[0].right,h+1])
#             max_key = max(max_key,h+1)
#     for key in range(min_key,max_key+1):
#         print(level_map[key], end=" ")
def bottomView( root):
    """ https://www.geeksforgeeks.org/problems/bottom-view-of-binary-tree/1
    modified vertical order traversal to overwrite new element in each level / last element in each vertical are part to bottom view"""
    res = []
    q = deque()
    q.append([root,0])
    verticals = dict()
    
    while len(q)>0:
        size = len(q)
        for _ in range(size):
            node, ver = q.popleft()
            verticals[ver] = node.data
            if node.left:
                q.append([node.left, ver-1])
            if node.right:
                q.append([node.right, ver+1])
    vers = sorted(verticals)
    
    for ver in vers:
        res.append(verticals[ver])
    return res

# def OLDbottom_view(root):
#     """ modified vertical order traversal to overwrite new element in each level """
#     level_map = dict()
#     q = deque()
#     q.append([root,0])
#     min_key = float('inf')
#     max_key = float('-inf')
#     while(len(q)):
#         data = q.popleft()
#         h = data[1]
#         if h not in level_map:
#             level_map[h] = None
#         level_map[h] = data[0].data  # this will be skipped for existing vertical
#         if data[0].left:
#             q.append([data[0].left,h-1])
#             min_key = min(min_key,h-1)
#         if data[0].right:
#             q.append([data[0].right,h+1])
#             max_key = max(max_key,h+1)
#     for key in range(min_key,max_key+1):
#         print(level_map[key], end=" ")

def left_and_right_view(root):
    res = []
    right_view_rec(root,0,res)
    print("Right view: ",res)
    res = []
    left_view_rec(root,0,res)
    print("Left view: ",res)

def  right_view_rec(root,level,res):
    """ Do preorder trav N-R-L, first elem in each level will be right view.
        Trick is to get size of result to check if it is visited first in each level"""
    if root is None:
        return
    if len(res) == level:
        res.append(root.data)
    if root.right:
        right_view_rec(root.right,level+1,res)
    if root.left:
        right_view_rec(root.left,level+1,res)

def left_view_rec(root,level,res):
    """ Do preorder trav N-L-R, first elem in  each level will be left view """
    if root is None:
        return
    if len(res) == level:
        res.append(root.data)
    if root.left:
        left_view_rec(root.left,level+1,res)
    if root.right:
        left_view_rec(root.right,level+1,res)

def rightSideView( root ):
    if root is None:
        return []
    q = deque()
    q.append(root)
    result = []
    while q:
        size = len(q)
        right = None
        for _ in range(size):
            temp = q.popleft()
            if right is None:
                right = temp.val
            if temp.right:
                q.append(temp.right)
            if temp.left:
                q.append(temp.left)
        result.append(right)
    return result

def is_mirror(left_ptr, right_ptr):
    if left_ptr is None or right_ptr is None:
        return left_ptr == right_ptr
    if left_ptr.data != right_ptr.data:
        return False
    return (is_mirror(left_ptr.left,right_ptr.right)
            and is_mirror(left_ptr.right,right_ptr.left))

def isSymmetric(root):
    """https://www.geeksforgeeks.org/problems/symmetric-tree/1"""
    def check(root1, root2):
        if root1 is None and root2 is None:
            return True
        if root1 is None or root2 is None:
            return False
        if root1.data != root2.data:
            return False
        leftside = check(root1.left, root2.right)
        rightside = check(root1.right, root2.left)
        return leftside and rightside
    
    if root is None:
        return True
    return check(root.left, root.right)

def path_to_element(root,path,k):
    if root is None:
        return False
    path.append(root.data)
    if k == root.data:
        return True
    found_left = path_to_element(root.left,path,k)
    if not found_left:
        found_right = path_to_element(root.right,path,k)
    if found_left or found_right:
        return True
    path.pop()
    return False

def binaryTreePaths(root):
    """https://leetcode.com/problems/binary-tree-paths/
       Print all the paths from root to leafs"""
    paths = []
    def create_path_str(path):
        res = ""
        for n in path:
            res += str(n)+"->"
        return res[:-2]
    
    def travel(node, path):
        path.append(node.val)
        if node.left is None and node.right is None:
            paths.append(create_path_str(path))
        if node.left:
            travel(node.left, path)
        if node.right:
            travel(node.right, path)
        path.pop()
    travel(root, [])
    return paths

class LowestCommonAncestor:
    """https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
    Given that solution always exists.
    IMPORTANT: For edge case where one of the p,q is itself ancestor of another: 
        let's suppose p happens to be ancestor of q then our recursive call will never reach q and finally return p which is valid solution.

    In postorder recursive call,  return p,q,none,or lca"""
    def lowestCommonAncestor(self, root, p, q):
        if root is None or root == p or root == q:
            return root
        leftside = self.lowestCommonAncestor(root.left, p, q)
        rightside = self.lowestCommonAncestor(root.right, p, q)
        if leftside is None:
            return rightside  # return : None, q, or lca
        if rightside is None:
            return leftside   # return: p or lca
        return root           # return: lca (found)

class Solution:
    """https://leetcode.com/problems/binary-tree-maximum-path-sum/solutions/4554272/python3-recursive-solution/"""
    def rec(self,node):
            if node is None:
                return 0
            l = self.rec(node.left)
            r = self.rec(node.right)

            max_here = max(node.val , node.val + max(l,r))
            self.res = max(self.res, max(max_here, l+r+node.val))
            
            return max_here
        
    def maxPathSum(self, root) -> int:
        self.res = float("-inf")
        self.rec(root)
        return self.res

class Diameter:
    
    def solve(self, root):
        if root is None:
            return 0
        left = self.solve(root.left)
        right = self.solve(root.right)
        self.res = max(self.res, 1 + left + right)

        return 1 + max(left,right)
    
    def diameter(self,root):
        """https://www.geeksforgeeks.org/problems/diameter-of-binary-tree/1
        https://leetcode.com/problems/diameter-of-binary-tree/description/"""
        self.res = 0
        self.solve(root)
        return self.res

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class CountNodesCompleteTree:
    """ Count nodes in complete binary tree in less than N time complexity.
    https://leetcode.com/problems/count-complete-tree-nodes/
    https://takeuforward.org/binary-tree/count-number-of-nodes-in-a-binary-tree/ """
    def countNodes(self, root) -> int:
        def left_h(node):
            c = 0
            while node:
                c += 1
                node = node.left
            return c

        def right_h(node):
            c = 0
            while node:
                c += 1
                node = node.right
            return c
        
        def count(node):
            if node is None:
                return 0

            l = left_h(node)
            r = right_h(node)
            if l == r:
                return pow(2,r)-1
            l = count(node.left)
            r = count(node.right)
            return 1 + l + r
        
        return count(root)


root = Node(10)
root.right = Node(30)
root.left = Node(20)
root.left.left = Node(40)
# root.left.left.left = Node(60)  # enable for unbalancing tree
root.left.right = Node(50)

"""
    10
  20  30
40 50
"""

root2 = Node(10)
root2.left = Node(30)
root2.right = Node(20)
root2.right.right = Node(40)
# root.left.left.left = Node(60)  # enable for unbalancing tree
root2.right.left = Node(50)

root3 = Node(10)
root3.left = Node(20)
root3.left.left = Node(30)
root3.left.left.right = Node(40)

root3.right = Node(20)
root3.right.right = Node(30)
root3.right.right.left = Node(40)
"""
    10
  20  20
30      30
  40  40
"""

prob = Problems()
print("\nLevel Order", end=" : ")
level_order(root)
print("\nLevel Order by each level 1", end=" : ")
level_order_by_level_1(root)
print("\nLevel Order by each level 2", end=" : ")
level_order_by_level_2(root)
print("\nPreorder",end=" : ")
print(preorder_iterative(root))
print("\nPost Order Iterative", end=" : ")
postorder_iterative(root)
print("\nIs Balanced: ",check_balance(root) >= 0 )
print("\nMax Width", end=" ")
prob.max_width(root)
print(prob.max_w)
print("Boundary Trav ", end=" : ")
print(boundary_trav(root))
print("Leaf Nodes",end=" : ")
arr = []
get_leafs(root,arr)
print(arr)
print("\nVertical Order", end=" : ")
vertical_order_traversal(root)
print("\nTop view",end=" : ")
topView(root)
print()
bottomView(root)
print()
left_and_right_view(root)
print("Is Mirror: ",is_mirror(root,root2))
print("Is Symmetric: ",is_mirror(root3.left,root3.right))
path = []
path_to_element(root,path,50)
print("Path to K", path )
lca = LowestCommonAncestor().lowestCommonAncestor(root, root3, root2)
print(lca.data)



