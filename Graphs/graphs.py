from collections import deque
from heapq import heappush, heappop
from typing import List

def bfs(v,adj_list, visited = None):
    output = []
    if visited is None:
        visited = [False]*len(adj_list)
    q = deque()
    q.append(v)
    visited[v] = True

    while len(q) > 0:
        node_i = q.popleft()
        output.append(node_i)
        for neigh in adj_list[node_i]:
            if not visited[neigh]:
                q.append(neigh)
                visited[neigh] = True

    return output

def dfs_rec(v, adj, visited, output):
    visited[v] = True
    output.append(v)
    for neigh in adj[v]:
        if not visited[neigh]:
            dfs_rec(neigh, adj, visited, output)
    return output

def dfs(adj):
    output = []
    visited = [False] * len(adj_list)
    dfs_rec(1,adj,visited,output)
    print(output)

def detect_cycle_undir_dfs(adj_list, visited, node, parent):
    """Approach: We use DFS to traverse the graph and keep track of visited nodes.
    If we find any visited node which is not the parent of current node, then it is a cycle (as it is undirected graph)
    Time Complexity: O(V+E)
    Space Complexity: O(V)"""
    visited[node] = True
    for adj in adj_list[node]:
        if not visited[node]:
            if detect_cycle_undir_dfs(adj_list, visited, adj, node): # if cycle found in any child node then return True
                return True
        elif node != parent:
            return True
    return False

def detect_cycle_undir_bfs(adj_list):
    """Approach: We use BFS to traverse the graph and keep track of visited nodes.
    If we find any visited node which is not the parent of current node, then it is a cycle (as it is undirected graph)
    Time Complexity: O(V+E)
    Space Complexity: O(V)
    """
    visited = [False]*(len(adj_list)+1)
    q = deque()
    q.append((1,-1)) # (current node, immediate parent)
    while len(q) > 0:
        node, par = q.popleft()
        for adj in adj_list[node]:
            if not visited[node]:
                q.append((adj, node))
                visited[adj] = True
            elif adj != par: # if the visited node is not the parent of current node, then it is a cycle (as it is undirected graph)
                return True
    return False

class detect_cycle_directed:
    """Time Complexity: O(V + E)
        Space Complexity: O(V)"""
    def check(self, node, adj,  visited, pvisited):
        visited[node] = True
        pvisited[node] = True
        
        # check all neigh
        for neigh in adj[node]:
            if not visited[neigh]:
                if self.check(neigh, adj, visited, pvisited):
                    # child node got cycle
                    return True
            elif pvisited[neigh]:
                    # current node got cycle
                return True
        
        pvisited[node] = False # unvisit node from pvisited
        return False
    
    #Function to detect cycle in a directed graph.
    def isCyclic(self, V, adj):
        visited = [False]*V     # normal visited array
        pvisited = [False]*V    # path visited array (maintains current path's only visited nodes)
        
        for v in range(V):
            if not visited[v]:
                if self.check(v, adj, visited, pvisited):   # as soon as it finds cycle returns
                    return True
        return False    # hence no cycle found

class eventual_safe_state:
    """https://www.geeksforgeeks.org/problems/eventual-safe-states/1"""
    def check(self, node, adj,  visited, pvisited):
        visited[node] = True
        pvisited[node] = True
        
        # check all neigh
        for neigh in adj[node]:
            if not visited[neigh]:
                if self.check(neigh, adj, visited, pvisited):
                    # child node got cycle -> mark current node unsafe
                    self.unsafe[node] = True
                    return True
            elif pvisited[neigh]:
                # current node got cycle -> mark unsafe
                self.unsafe[node] = True
                return True
        
        pvisited[node] = False
        return False
        
    def eventualSafeNodes(self, V : int, adj : List[List[int]]) -> List[int]:
        # If node leads to cycle then it is unsafe
        visited = [False]*V
        pvisited = [False]*V
        
        safe, self.unsafe = [], [False]*V
        
        for v in range(V):
            if not visited[v]:
                self.check(v, adj, visited, pvisited)
                
        for v in range(V):
            if not self.unsafe[v]:
                safe.append(v)
        return safe

class IsBipartiteProblem:
    def bfs(self, node_i, graph, vis):
        """In BFS we simply modified the visited array to store the color of the node instead of just boolean value. 
        -1 for unvisited, 0 for color A, 1 for color B. 
        Then we traverse the graph and if we find any neighbour with same color as current node, then it is not bipartite."""
        q = deque()
        q.append(node_i)
        while len(q) > 0:
            node = q.popleft()
            for adj in graph[node]:
                if vis[adj] < 0:
                    vis[adj] = 0 if vis[node] == 1 else 1
                    q.append(adj)
                elif vis[adj] == vis[node]:
                    return False
        return True
    
    def dfs(self, node, graph, vis, color):
        """In DFS we simply modified the visited array to store the color of the node instead of just boolean value.
        -1 for unvisited, 0 for color A, 1 for color B.
        Then we traverse in DFS and pass new color to each child node. If we find any neighbour with same color as current node, then it is not bipartite."""
        vis[node] = color
        for adj in graph[node]:
            if vis[adj] < 0:
                if not self.dfs(adj, graph, vis, 0 if color == 1 else 1):
                    return False
            elif vis[adj] == color:
                return False
        return True

    def isBipartite(self, graph):
        vis = [-1 for _ in range(len(graph))]
        
        for i in range(len(graph)):
            if vis[i] == -1:
                vis[i] = 0
                # if not self.bfs(i, graph, vis, vis[0]):
                #     return False
                if not self.dfs(i, graph, vis, 0):
                    return False
        return True

def detect_cycle_undir(adj_list):

    visited = [False]*(len(adj_list)+1)
    return detect_cycle_undir_bfs(adj_list)
    return detect_cycle_undir_dfs(adj_list,visited, 1, -1)

#==========================================================
def dfs_topo(node, order, vis, adj_list):
    """DFS based topological sort
    Time Complexity: O(V+E)
    Space Complexity: O(V)"""
    vis[node] = True
    for adj in adj_list[node]: 
        if not vis[adj]:
            dfs_topo(adj, order, vis, adj_list)

    order.append(node)  # the only modified part of DFS ->add a element to stack after reaching depth

def topo_all(adj_list):
    n = len(adj_list)
    vis = [False]*n
    indeg = [0]*n
    for src in range(n):
        for child in adj_list[src]:
            indeg[child] += 1
    output = []
    def dfs(path):
        for node in range(n):
            if vis[node] or indeg[node] != 0:
                continue
            vis[node] = True
            for child in adj_list[node]:
                if not vis[child]:
                    indeg[child] -= 1
            path.append(node)

            dfs(path)

            vis[node] = False
            for child in adj_list[node]:
                indeg[child] += 1
            if len(path) == n:
                output.append(path.copy()[::-1])
            path.pop()
    dfs([])
    print(output)
 
def kahn_bfs_topo(adj_list):
    # Uses Indegree calculation and starts relaxing edges in BFS manner starting from 0 In-degree nodes
    n = len(adj_list)
    indeg = [0 for _ in range(n)]
    for node in range(n):
        for adj in adj_list[node]:
            indeg[adj] += 1

    # enque all nodes with 0 indegree to init queue and do BFS
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)
    order = []
    while len(q):
        node = q.popleft()
        order.append(node)
        for adj in adj_list[node]:
            indeg[adj] -= 1
            if indeg[adj] == 0:  # only if indegree becomes 0 of any neighbour we can enque it to Q
                q.append(adj)
    return order

def topological_sort(adj_list):
    vis = [False for _ in range(len(adj_list))]
    order = []
    for node in range(len(adj_list)):
        if not vis[node]:
            dfs_topo(node, order, vis, adj_list)
    return order[::-1]   # reverse the stack

#===========================================
def dijkstra(adj, src : int):
    # Shortest path from src to all other nodes (directed & undirected), works for non-negative edge only
    # O(E*log(V))
    """Approach: We use Dijkstra's algorithm to find the shortest path from source to all other nodes.
    We use a priority queue to store the nodes and their distance from the source. (distance, node)
    We keep popping the node with minimum distance from the priority queue and relax its neighbours.
    Dijkstra's algorithm works for both directed and undirected graphs. But for negative edge weights, we use Bellman Ford.

    Time Complexity: O(V + E*log(V))
    Space Complexity: O(V)   for distance array
    """
    pq = [(0,src)]
    dist = [float("inf")]*len(adj)
    dist[src] = 0
    
    while len(pq):
        d, node = heappop(pq)
        for neigh, w in adj[node]:
            if w + d < dist[neigh]: # new distance is less than previous (if not calculated then it will be INF)
                dist[neigh] = w + d # update the distance w is the weight of edge and d is the distance of current node
                heappush(pq,(dist[neigh],neigh)) # add the new distance to PQ
    
    return dist
    # https://leetcode.com/problems/cheapest-flights-within-k-stops/
    
def bellman_ford( V, edges, S):
    """
    Detects negative edge cycle, Single sourced shortest path in Directed & Undirected graphs
    Approach: We use Bellman Ford algorithm to find the shortest path from source to all other nodes.

    We use a list to store the distance from the source to all other nodes. We initialize the distance of source to 0 and all other nodes to infinity.
    Step 1: We then relax all the edges V-1 times. If we find any distance less than the current distance, then we update the distance.
    Step 2: If we find any distance less than the current distance after V-1 iterations, then it means there is a negative cycle in a graph.
    
    Negative cycle means that the sum of the weights of the edges in the cycle is negative hence we can keep going in the cycle to decrease the distance to negative infinity.
    
    Time complexity: O(V*E)
    """
    inf = float("inf")
    dist = [inf]*V
    dist[S] = 0
    for _ in range(V-1):
        for edge in edges:
            u, v, w = edge  # u->v with weight w
            # (Discarded code)if dist[u] is not inf and dist[v] > dist[u] + w:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for edge in edges:
            u, v, w = edge
            if dist[u] + w < dist[v]:
                return [-1] # negative cycle detected/ return whatever problem statement asks to return
    return dist

def floyd_warshall( matrix ):
    """Floyd Warshall Algorithm to find shortest path between all pairs of nodes.
    Time Complexity: O(V^3)
    Space Complexity: O(V^2)

    Approach: We use Dynamic Programming to find the shortest path between all pairs of nodes.
    We first clean the data and then use 3 nested loops to find the shortest path between all pairs of nodes.
    We then replace the INF with -1 for output.

    Core Idea: for every edge A-> B, recalculate path sum N time comparing  "path(A->k) + path(k->B)" with path(A->B)
    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])"""
    # Path sourced shortest path algo./ Uses Dynamic programming / O(V*V*V)
    # in input -1 means no edge.
    # Step 1,3 are just for helpers, main algorithm is Step 2 only

    n = len(matrix)
    inf = float("inf")

    # Step 1: clean data
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0 # distance to itself is 0
            elif matrix[i][j] == -1:
                matrix[i][j] = inf # input -1 means no edge so replace with infinity

    # Step 2: For every node k, check all pairs of nodes i and j and update the shortest path between i and j through k
    for k in range(n): # k is the intermediate node
        for i in range(n): # i is the source node
            for j in range(n): # j is the destination node

                path = matrix[i][k] + matrix[k][j] # path from i to j through k is the sum of path from i to k and k to j
                matrix[i][j] = min(matrix[i][j], path)

    # step 3: replacing INF with -1 for output
    for i in range(n):
        for j in range(n):
            if matrix[i][j] >= inf:
                matrix[i][j] = -1


def prims_MST( V, adj ):
    """Approach: We use Prim's algorithm to find the minimum spanning tree of the graph.
    It is a greedy algorithm. It works for both directed and undirected graphs.

    Prims algorithm is similar to Dijkstra's algorithm. 
    The only difference is that in Dijkstra's algorithm we keep track of the shortest distance from the source to all other nodes,
    while in Prim's algorithm we keep track of the shortest distance from the source to all other nodes in the minimum spanning tree.

    (IMP)  We use a priority queue to store the nodes and their distance from the parent. (weight, node, edge)
    We keep popping the node with minimum distance from the priority queue and relax its neighbours.
    Time Complexity: O(E*log(V))
    Space Complexity: O(E)
    Core Idea: Start from any node, add all unvisited neighbours to PQ, PQ will handle pop of minimum weighted edge in next iteration."""

    vis = [False for _ in range(V)]
    pq = [(0,0,None)] # weight, node
    
    mst_weight_sum = 0
    mst_edges = []
    while len(pq):
        node_w, node, edge = heappop(pq)
        if vis[node]:  # As it is PQ, someone could have already visited after it was added to PQ, so we  check visited[node] again
            continue
        mst_weight_sum += node_w
        vis[node] = True

        if edge:
            mst_edges.append(edge)

        # add all unvisited neighbours to PQ, PQ will handle pop of minimum weighted edge in next iteration.
        for neigh, w in adj[node]:
            if not vis[neigh]:
                heappush(pq, (w, neigh, (node, neigh)))

    return mst_weight_sum, mst_edges

class disjoint_set:
    """Disjoint Set Data Structure with Union by Rank and Path Compression
    Rank means the height of the tree. We always attach the smaller tree to the root of the larger tree.
    Path Compression means that we make the parent of the node as the root of the tree while finding the parent of the node.
    Time Complexity: O(1) for find and O(1) for union on average.
    Space Complexity: O(V)

    Description: Disjoint Set Data Structure is a data structure that keeps track of a set of elements partitioned 
    into a number of disjoint (non-overlapping) subsets.
    It supports two useful operations:
    1. Find: Determine which subset a particular element is in. This can be used for determining if two elements are in the same subset.
    2. Union: Join two subsets into a single subset. Types: Union by Rank and Path Compression (Optimized) and Union by Size (Optimized)
    """
    def __init__(self, n) -> None:
        self.parent = [i for i in range(n)] # initially every one is parent of itself
        self.rank = [0]*n   # 
        self.size = [1]*n   # size of every individual group is 1
    
    def ul_parent(self, node):
        if self.parent[node] == node:
            return node
        top_parent = self.ul_parent(self.parent[node])
        self.parent[node] = top_parent
        return top_parent
    
    def union_rank(self, u, v):
        ulp_u = self.ul_parent(u)
        ulp_v = self.ul_parent(v)
        if ulp_u == ulp_v:
            return
        elif self.rank[ulp_u] < self.rank[ulp_v]:
            self.parent[ulp_u] = ulp_v
        elif self.rank[ulp_u] > self.rank[ulp_v]:
            self.parent[ulp_v] = ulp_u
        else: # if both rank equal
            self.parent[ulp_v] = ulp_u
            self.rank[ulp_u] += 1
    
    def union_size(self, u, v): # time complexity O(logn) on average (amortized time complexity is O(1) on average
        
        ulp_u = self.ul_parent(u)
        ulp_v = self.ul_parent(v)

        if ulp_u == ulp_v: # if already in union/ same parent
            return
        # attach smaller tree to larger tree
        elif self.size[ulp_u] < self.size[ulp_v]:
            self.parent[ulp_u] = ulp_v
            self.size[ulp_v] += self.size[ulp_u]
        else:
            self.parent[ulp_v] = ulp_u
            self.size[ulp_u] += self.size[ulp_v]

    def find(self, u, v):
        return self.ul_parent(u) == self.ul_parent(v)

    def run(self,bysize=False):

        union = self.union_size if bysize else self.union_rank
        union(0,1)
        union(1,2)
        union(3,4)
        union(5,6)
        union(4,5)
        union(2,6)

        # This will ensure that all path compression possible are done.
        for _ in range(7):
            print(self.ul_parent(_))

dj = disjoint_set(7)
# dj.run(bysize=True)
# print("rank   ",dj.rank)
# print("size   ",dj.size)
# print("parent ",dj.parent)

class Kruskal_MST:

    def ul_parent(self, node):
        if node == self.parent[node]:
            return node
        top_parent = self.ul_parent(self.parent[node])
        self.parent[node] = top_parent
        return top_parent
    
    def union(self, u, v):
        up = self.ul_parent(u)
        vp = self.ul_parent(v)
        
        if up == vp:
            return
        elif self.rank[up] < self.rank[vp]:
            self.parent[up] = vp
        elif self.rank[vp] < self.rank[up]:
            self.parent[vp] = up
        else:
            self.parent[vp] = up
            self.rank[up] += 1
    
    def spanningTree(self, V, adj):
        #code here
        self.rank = [0]*V
        self.parent = [ _ for _ in range(V)]   #Initialised with parent of itself
        
        total = 0
        edges = []
        for node in range(V):
            for link in adj[node]:
                edges.append( (link[1], node, link[0]) )   # (weight, u, v)

        edges.sort(key=lambda x: x[0])

        for edge in edges:
            if self.ul_parent(edge[1]) != self.ul_parent(edge[2]):
                    total += edge[0]
                    self.union(edge[1],edge[2])

        return total

class KosarajuSCC:
    """Algo to find number of strongly connected components in graph.
       A component is called Strongly connected if all nodes are reachable from each other as starting point.
       
       Time compl = O(V+E)"""

    def dfs(self, node, adj, stk, vis):
        vis[node] = True
        for nxt in adj[node]:
            if not vis[nxt]:
                self.dfs(nxt, adj, stk, vis)
        stk.append(node)
    
    def dfs2(self, node, adj, vis):
        vis[node] = True
        for nxt in adj[node]:
            if not vis[nxt]:
                self.dfs2(nxt, adj, vis)
        
    #Function to find number of strongly connected components in the graph.
    def kosaraju(self, V, adj):
        """Approach: We use Kosaraju's algorithm to find the number of strongly connected components in the graph.
        1. We first traverse the graph and store the nodes in stack in order of their finish time(order: DFS leaf to root)
        2. Then we reverse the graph and traverse the nodes in stack and count the number of times we reach a new node.
        3. Do DFS created from stack: The count of number of times we reach a new node is the number of strongly connected components in the graph.
        Time Complexity: O(V+E)
        Note: We can also print the strongly connected components using the same approach."""
        vis = [False]*V
        stk = []    # to save graph in original order
        
        # get original order in stack
        for v in range(V):
            if not vis[v]:
                self.dfs(v,adj,stk,vis)
        
        # reverse the graph into radj
        radj = [ [] for v in range(V)]
        for v in range(V):
            for nxt in adj[v]:
                radj[nxt].append(v)
        
        scc = 0
        vis = [False]*V

        # call dfs in reversed graph (radj) but in original order using stack
        while stk:
            node = stk.pop()
            if not vis[node]:
                self.dfs2(node, radj, vis)
                scc += 1
        
        return scc

if __name__ == "__main__":
    adj_list = [[],[2,6],[1,3,4,7],[2],[2,5],[4,7],[1],[2,5]]
    print(bfs(1, adj_list))
    dfs(adj_list)

    print("cycle: ", detect_cycle_undir( adj_list ))

    adj_list_disconnected = [[],[2,3],[1],[1],[5],[4],[]]

    adj_matrix = [[0,1,1,0],
              [0,1,1,0],
              [0,0,1,0],
              [0,0,0,0],
              [1,1,0,1]]
    directed_adj_list = [[], [], [3], [1], [0,1], [0,2]]
    #print("Topological sort (DFS): ",topological_sort( directed_adj_list ))
    #print("Topological sort Kahn (BFS): ",kahn_bfs_topo(directed_adj_list))
    topo_all(directed_adj_list)

# m,n = 3,6
# vis = [[False for i in range(n)] for j in range(m)]
# print(vis)