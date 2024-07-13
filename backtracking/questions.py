from typing import List


class StrPermutate:

    def recursion(self, ip, o):
        """ Time complexity:  O(N!)"""
        if len(ip) == 0:
            self.res.append(o)
            return
        for i in range(len(ip)):
            if ip[i] in self.mem:
                continue
            ip_n = ip[:i] + ip[i+1:]
            o_n = o + ip[i]
            self.mem.add(ip[i])
            self.recursion( ip_n, o_n)

    def backtrack(self, ip: str, op: str):
        """O(N^2)"""
        if len(ip) == 0:
            self.res.append(op)
            return
        for i in range(len(ip)):
            op += ip[i]
            ip_n = ip[:i]+ip[i+1:]
            self.backtrack( ip_n, op)
            op = op[:-1]


    def permutate(self, s):
        self.res = []
        self.mem = set()
        self.recursion(s, "")

        # self.backtrack(s, "")
        print(self.res)

StrPermutate().permutate("ABC")


class NQueens:
    """https://leetcode.com/problems/n-queens/"""
    def solveNQueens(self, n: int) -> List[List[str]]:
        rows, cols, diag, adiag = set(), set(), set(), set()
        grids = []
        def solve(r, grid):

            if r == n:
                plane_grid = []
                for row in grid:
                    single_row = "".join(ci for ci in row)
                    plane_grid.append(single_row)
                grids.append(plane_grid)
                return

            for ci in range(n):
                possible = True
                if r in rows or ci in cols:
                    possible = False
                if ci-r in diag or  r+ci in adiag:
                    possible = False
                if not possible:
                    continue

                rows.add(r)
                cols.add(ci)
                diag.add(ci-r)
                adiag.add(ci+r)
                grid[r][ci] = "Q"

                solve(r+1, grid) # next row

                rows.remove(r)
                cols.remove(ci)
                diag.remove(ci-r)
                adiag.remove(ci+r)
                
                grid[r][ci] = "."

            return
        
        grid = [["."]*n for _ in range(n)]
        solve(0, grid)
        return grids

class NQueen2:
    """https://leetcode.com/problems/n-queens-ii/
    The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
    Given an integer n, return the number of distinct solutions to the n-queens puzzle."""
    def totalNQueens(self, n: int) -> int:
        diag = set() # diff
        adiag = set() # sum
        xset, yset = set(), set()

        def possible(x,y):
            if x in xset:
                return False
            if y in yset:
                return False
            if x+y in adiag:
                return False
            if y-x in diag:
                return False
            return True
        
        def place(x,y):
            xset.add(x)
            yset.add(y)
            diag.add(y-x)
            adiag.add(x+y)

        def pick(x,y):
            xset.remove(x)
            yset.remove(y)
            diag.remove(y-x)
            adiag.remove(x+y)

        def solve(r,n):
            if r >= n: # if completed board
                return 1
            c = 0 # try all columns indexes
            ways = 0
            while c < n:
                if possible(r,c):
                    place(r,c)
                    ways += solve(r+1,n)
                    pick(r,c)
                c += 1
            return ways
        return solve(0,n)


class Permutate:
    def permutate(self, n: int, k: int) -> List[List[int]]:

        result = []

        def solve(perm, unpicked):
            if len(perm) == k:
                result.append(perm.copy())
                return
            for d in range(1,n+1):
                if unpicked[d]:
                    perm.append(d)
                    unpicked[d] = False
                    solve(perm, unpicked)
                    unpicked[d] = True
                    perm.pop()
        
        unpicked = [True]*(n+1)
        solve([],unpicked)
        return result

class Combination:
    """https://leetcode.com/problems/combinations/solutions/4747292/recursive-solution-o-nck-solution/
    Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].
    """
    def combine(self, n: int, k: int) -> List[List[int]]:

        result = []

        def solve(comb, d):
            if len(comb) == k:
                result.append(comb.copy())
            elif d <= n:
                comb.append(d)
                solve(comb, d+1)

                comb.pop()
                solve(comb, d+1)

        solve([],1)
        return result
    
class Problem24Game:
    """https://leetcode.com/problems/24-game/
See get_vals() function. It shows 6 possible calculations between two numbers A and B with given operators.

Create recursive function which takes array/list input cards.
We pick two numbers A and B from cards and generate all possible values usisng get_vals and append it with remaining cards (other than A and B) into new array/list new_cards for another recursive call to same function judgePoint24()
BASE CONDITION: if length of cards is 1 at any recursive call we can return either True or False according to the value in cards.
If value is 24 return True (ignore decimal values less than 0.1)
else return False"""
    def get_vals(self, a, b):
        vals = [a+b, a-b, b-a, a*b]
        if a!=0:
            vals.append(b/a)
        if b!=0:
            vals.append(a/b)
        return vals

    def judgePoint24(self, cards: List[int]) -> bool:
        if len(cards) == 1:
            if abs(24.0 - cards[-1]) < 0.1:
                return True
            return False
        ans = False
        n = len(cards)
        for i in range(n):
            a = cards[i]
            for j in range(i+1,n):
                b = cards[j]
                vals = self.get_vals(a,b)
                for vi in range(len(vals)):
                    val = vals[vi]
                    new_cards = [card for k,card in enumerate(cards) if k!=i and k!=j]
                    new_cards.append(val)
                    ans = ans or self.judgePoint24(new_cards)
                    if ans:
                        return True

        return ans
    
    def run(self):
        print(self.judgePoint24([4,1,8,7])) # True
        print(self.judgePoint24([1,2,1,2])) # False