from collections import deque
from heapq import heappush, heappop, heapify

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
    visited[node] = True
    for adj in adj_list[node]:
        if not visited[node]:
            if detect_cycle_undir_dfs(adj_list, visited, adj, node):
                return True
        elif node != parent:
            return True
    return False

def detect_cycle_undir_bfs(adj_list):
    visited = [False]*(len(adj_list)+1)
    q = deque()
    q.append((1,-1)) # (current node, immediate parent)
    while len(q) > 0:
        node = q.popleft()
        for adj in adj_list[node[0]]:
            if not visited[node[0]]:
                q.append((adj,node[0]))
                visited[adj] = True
            elif adj != node[1]:
                return True
    return False

class IsBipartiteProblem:
    def bfs(self, node_i, graph, vis, color):
        q = deque()
        q.append(node_i)
        while len(q) > 0:
            node = q.popleft()
            for adj in graph[node]:
                if vis[adj] < 0:
                    vis[adj] = self.switch[vis[node]]
                    q.append(adj)
                elif vis[adj] == vis[node]:
                    return False
        return True
    
    def dfs(self, node, graph, vis, color):
        vis[node] = color
        for adj in graph[node]:
            if vis[adj] < 0:
                if not self.dfs(adj, graph, vis, self.switch[color]):
                    return False
            elif vis[adj] == color:
                return False
        return True

    def isBipartite(self, graph):
        vis = [-1 for _ in range(len(graph))]
        self.switch = [1,0]
        #two color 0/1
        
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
    vis[node] = True
    for adj in adj_list[node]: 
        if not vis[adj]:
            dfs_topo(adj,order,vis,adj_list)

    order.append(node)  # the only modified part of DFS ->adda element to stack after reaching depth
 
def kahn_bfs_topo(adj_list):
    # Uses Indegree calculation and starts relaxing edges in BFS manner starting from 0 Indegegree nodes
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
def dijkstra(adj, src):
        # Shortest path from src to all other nodes, works for non-negative edge only
        # O(E*log(V))
        pq = [(0,src)]
        dist = [None]*len(adj)
        dist[src] = 0
        
        while len(pq):
            d,node = heappop(pq)
            for neigh,w in adj[node]:
                if dist[neigh] is None or dist[neigh]> w + d:
                    dist[neigh] = w + d
                    heappush(pq,(dist[neigh],neigh))
        
        return dist

def bellman_ford( V, edges, S):
    # Detects negative edge, Single sourced shortest path, O(V*E)
    inf = 100000000
    dist = [inf]*V
    dist[S] = 0
    for i in range(V-1):
        for edge in edges:
            u, v, w = edge
            if dist[u] is not inf and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
    for edge in edges:
            u, v, w = edge
            if dist[u] is not inf and dist[v] > dist[u] + w:
                return [-1]
    return dist

def floyd_warshall( matrix ):
    # Path sourced shortest path algo./ Uses Dynamic programming / O(V*V*V)
    # in input -1 means no edge.
    # Step 1,3 are just for helpers, main algorith is Step 2 only

    n = len(matrix)
    inf = float("inf")

    # Step 1: clean data
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            elif matrix[i][j] == -1:
                matrix[i][j] = inf
    # Step 2: for every edge A-> B, recalc path sum N time comparing  "path(A->k) + path(k->B)" with path(A->B)
    for k in range(n):
        for i in range(n):
            for j in range(n):

                path = matrix[i][k] + matrix[k][j]
                if path < matrix[i][j]:
                    matrix[i][j] = path

    # step 3: replacing INF with -1 for output
    for i in range(n):
        for j in range(n):
            if matrix[i][j] >= inf:
                matrix[i][j] = -1


def prims_MST( V, adj):
    # O(E*log(E))  space O(E)
    vis = [False for _ in range(V)]
    pq = [(0,0,None)]
    
    mst_weight_sum = 0
    mst_edges = []  # Optional - incase you need to show all edges in MST
    while len(pq):
        node_w, node, edge = heappop(pq)
        if vis[node]:  # As it is pqueue, someone could have already visited after it was added to q -> so check visited arr again
            continue
        mst_weight_sum += node_w
        vis[node] = True
        if edge:
            mst_edges.append(edge)

        for neigh, w in adj[node]:
            if not vis[neigh]:   # just add all unvisited neighbours to PQ, PQ will handle pop of minimum weighted edge in next iteration.
                heappush(pq, (w, neigh, (node,neigh)))
    return mst_weight_sum

class KruskalMST:
    """https://www.geeksforgeeks.org/problems/minimum-spanning-tree/1"""
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
    #Function to find sum of weights of edges of the Minimum Spanning Tree.
    def spanningTree(self, V, adj):
        #code here
        self.rank = [0]*V
        self.parent = [ _ for _ in range(V)]
        
        total = 0
        
        edges = []
        for node in range(V):
            for link in adj[node]:
                edges.append( (link[1], node, link[0]) )

        edges.sort(key=lambda x: x[0]) # sort

        for edge in edges:
            if self.ul_parent(edge[1]) != self.ul_parent(edge[2]):
                    total += edge[0]
                    self.union(edge[1],edge[2])

        return total


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
    print("Topological sort (DFS): ",topological_sort( directed_adj_list ))
    print("Topological sort Kahn (BFS): ",kahn_bfs_topo(directed_adj_list))

# m,n = 3,6
# vis = [[False for i in range(n)] for j in range(m)]
# print(vis)