from collections import deque
from graphs import bfs

def provinces_count(adj):
    province_count = 0
    visited = [False]*len(adj)
    for node in range(1,len(adj)):
        if not visited[node]:
            bfs(node,adj, visited)
            province_count += 1
    return province_count

def islands_count(graph):
    del_r = [0,-1,0,1]
    del_c = [1,0,-1,0]

    row_count = len(graph)
    col_count = len(graph[0])

    visited = [[False for i in range(len(graph[0]))] for j in range(len(graph))]
    island_count = 0

    for r in range(row_count):
        for c in range(col_count):
            if not visited[r][c] and graph[r][c] == 1:
                q = deque()
                q.append((r, c))
                while len(q) > 0:
                    node = q.popleft()
                    if not visited[node[0]][node[1]] and graph[node[0]][node[1]] == 1:
                        visited[node[0]][node[1]] = True

                        for k in range(4):
                            next_r = node[0] + del_r[k]
                            next_c = node[1] + del_c[k]
                            if 0 <= next_r < row_count and 0 <= next_c< col_count:
                                if graph[next_r][next_c] == 1:
                                    q.append((next_r, next_c))
                island_count += 1
    return island_count

def floodFill(image, sr: int, sc: int, color: int):
    del_r = [1,0,-1,0]
    del_c = [0,-1,0,1]

    r_count = len(image)
    c_count = len(image[0])

    q = deque()
    original_color = image[sr][sc]
    q.append((sr,sc))
    while len(q) > 0:
        pixel = q.popleft()
        image[pixel[0]][pixel[1]] = color

        for k in range(4):
            next_pixel_r = pixel[0] + del_r[k]
            next_pixel_c = pixel[1] + del_c[k]

            if 0 <= next_pixel_r < r_count and 0 <= next_pixel_c < c_count:
                if image[next_pixel_r][next_pixel_c] == original_color:
                    if image[next_pixel_r][next_pixel_c] != color:
                        q.append( (next_pixel_r, next_pixel_c) )
                        image[next_pixel_r][next_pixel_c] = color
    return image

## Rotten Tomatoes

def nearest01(mat):
    delr = [1,0,-1,0]
    delc = [0,-1,0,1]
    m = len(mat)
    n = len(mat[0])
    vis = [[False for i in range(n)] for j in range(m)]
    dis = [[0 for i in range(n)] for j in range(m)]
    q = deque()
    for r in range(m):
        for c in range(n):
            if mat[r][c] == 0:
                q.append( ((r,c),0) )
                vis[r][c] = True
    while len(q)>0:
        cell = q.popleft()
        step = cell[1]
        cord = cell[0]
        dis[cord[0]][cord[1]] = step
        for i in range(4):
            r,c = cord[0]+delr[i], cord[1]+delc[i]
            if 0<=r<m and 0<=c<n and not vis[r][c]:
                q.append(((r,c), step+1) )
                vis[r][c] = True
    return dis

class ShortestDistanceToAllFromSrc_DAG_weighted:
    """https://practice.geeksforgeeks.org/problems/shortest-path-in-undirected-graph/1"""
    def dfs_topo(self, node, graph):
        
        for adj in graph[node]:
            if not self.vis[adj[0]]:
                self.vis[adj[0]] = True
                self.dfs_topo(adj[0], graph)
        self.order.append(node)

    def shortestPath(self, n : int, m : int, edges ):
        self.vis = [False for _ in range(n)]
        dist = [None for _ in range(n)]
        src = 0
        
        adj_list = [[] for _ in range(n)]
        for edge in edges:
            adj_list[edge[0]].append( (edge[1], edge[2]) )
        
        self.order = []
        self.vis[src] = True
        self.dfs_topo(src, adj_list)
        order = self.order
        
        dist[src] = 0
        while len(order):
            node = order.pop()
            for adj,w in adj_list[node]:
                if not self.vis[adj]:
                    dist[adj] = -1
                elif dist[adj] is None:
                    dist[adj] = w + dist[node]
                else:
                    dist[adj] = min(dist[adj], w + dist[node])

        for node in range(n):
            dist[node] = -1 if not self.vis[node] else dist[node] 
        return dist

class ShortestDistanceToAllFromSrc_undirect_unitweight:
    """https://practice.geeksforgeeks.org/problems/shortest-path-in-undirected-graph-having-unit-distance/1"""

    def shortestPath(self, edges, n, m, src):
        dis = [ -1 for _ in range(n)] # output for distance to all nodes
        
        # create adjacency list from edge input
        adj_list = [ [] for _ in range(n) ]
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        # Init Queue with source node with distance 0  (node,distance from src)
        q = deque()
        q.append((src,0))
        dis[src] = 0
        
        while len(q):
            
            node, dis_p = q.popleft()
            for adj in adj_list[node]:
                if dis[adj] == -1:
                    dis[adj] = dis_p + 1
                    q.append((adj,dis[adj]))
        return dis


adj_list = [[],[2,6],[1,3,4,7],[2],[2,5],[4,7],[1],[2,5]]
adj_list_disconnected = [[],[2,3],[1],[1],[5],[4],[]]

print("province count: ",provinces_count(adj_list_disconnected))

adj_matrix = [[0,1,1,1],
              [0,1,1,0],
              [0,0,1,0],
              [0,0,0,0],
              [1,1,0,1]]
print("island count: ",islands_count(adj_matrix))
print("flood fill: ",floodFill(adj_matrix,0,1,2))
print("01 Matrix Leetcode: ",nearest01(adj_matrix))