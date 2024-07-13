

class Segmenttree:
    """ Segment Tree is to be used when we need to do M queries on N sized array,
    where query can update elements in array and can do range queries.
    Both operations takes O(logN) time """

    def __init__(self, N) -> None:
        self.N = N
        self.tree = [None]*(4*N)

    def build(self, input, node, s, e):
        "Post order"
        if s == e:
            self.tree[node] = input[s]
            return
        mid = s + (e-s)//2
        self.build(input, 2*node+1, s, mid)
        self.build(input, 2*node+2, mid+1,e)
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def query(self, s, e ):
        pass


seg = Segmenttree(8)
input = [5,3,2,4,1,8,6,10]
seg.build(input,0,0,len(input)-1)
print(seg.tree)
input.popleft()
n = 6
print(list(range(0,n,2)))