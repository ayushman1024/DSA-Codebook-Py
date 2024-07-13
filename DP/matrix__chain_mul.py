class MCM:
    def recursion(self,i,j,arr):
        g_min = float("inf")
        if i >= j:
            return 0
        
        # k: i to j-1 , k signifies that we checks to partition after text[k]
        # Therefore we iterate loop till j-1 only(even if it is 0 based indexed)
        """We are partitioning the array into two parts and then we are finding the minimum cost of multiplication of matrices
        from i to k and k+1 to j. We are doing this for all the k from i to j-1 and then we are finding the minimum cost
        of multiplication of matrices from i to j. We are doing this for all the i and j and then we are finding the minimum"""
        for k in range(i,j): # 0,1,2,3  0,1 1,2  2,3    l=0  r =n-1  p = l+1 to r-1
            left = self.rec(i,k,arr)
            right = self.rec(k+1,j,arr)

            mul = arr[i-1]*arr[k]*arr[j]
            res = mul + left + right
            g_min = min(g_min, res)
        return g_min

    def memoisation(self,i,j,arr):
        g_min = float("inf")
        if i >= j: # if i == j then there is only one matrix and we can't multiply it with any other matrix
            return 0
        if self.dp[i][j] > 0:   # if dp[i][j] is already calculated then return it
            return self.dp[i][j]
        for k in range(i,j):
            self.dp[i][k] = self.memoisation(i,k,arr)
            self.dp[k+1][j] = self.memoisation(k+1,j,arr)

            mul = arr[i-1]*arr[k]*arr[j] # cost of multiplication of matrices from i to k and k+1 to j
            res = mul + self.dp[i][k] + self.dp[k+1][j] 
            g_min = min(g_min, res)
        self.dp[i][j] = g_min
        return g_min
        
    def matrixMultiplication(self, N, arr):
        """
        what it does? It returns the minimum number of multiplications needed to multiply the matrices
        arr=[40,20,30,10,30] represents dimensions for 4 matrices, that are 40x20, 20x30, 30x10, 10x30
        also cost of multiplying A*B and B*C is  A*B*C . For eg. 40x20 and 20x30 is 40*20*30 (total number of multiplications).
        So, we need to find the minimum cost/no. of multiplication of all the matrices since the order of multiplication matters.
        """
        i, j = 1, len(arr)-1
        self.dp = [[0 for i in range(N)] for j in range(N)] #dp[i][j] means minimum cost of multiplication of matrices from i to j 
        # print(self.topdown(arr))
        return self.memoisation(i, j,arr)

mcm = MCM()
t = mcm.matrixMultiplication(5, [40,20,30,10,30])

class CutsToMakeAllSubsetsPalindrome:
    def is_palindrome(self,i,j,text):
        while i <= j:
            if text[i] != text[j]:
                return False
            i += 1
            j -= 1
        return True

    def rec(self, i,j, text):
        if i >= j:
            return 0
        elif self.dp[i][j] != -1:
            return self.dp[i][j]
        elif self.is_palindrome(i,j,text):
            self.dp[i][j] = 0
            return self.dp[i][j]
        
        g = float("inf")
        """k: i to j-1 , k signifies that we checks to partition after text[k]
             Therefore we iterate loop till j-1 only(even if it is 0 based indexed)"""
        for k in range(i,j):
            left = self.rec(i, k, text)
            right = self.rec(k+1, j, text)
            res = 1 + left + right
            if res < g:
                g = res
        self.dp[i][j] = g
        return g
    
    def memo(self,i, j, text):
        
        if i >= j:
            return 0
        elif self.dp[i][j] >= 0:
            return self.dp[i][j]
        elif self.is_palindrome(i,j,text):
            self.dp[i][j] = 0
            return 0
        else:
            g_min = float("inf")
            for k in range(i,j):
                left = self.memo(i,k,text)
                right = self.memo(k+1,j,text)
                calc = 1 + left + right
                if calc <g_min:
                    g_min = calc
            self.dp[i][j] = g_min
        return self.dp[i][j]
    
    def topdown(self, text):
        n = len(text)
        dp = [float("inf") for i in range(n)]
        
        for j in range(n):
            dp[j] = float("inf")
            for i in range(j,-1,-1):
                if self.is_palindrome(i,j,text):
                    if i == 0: # If it is complete substring
                        dp[j] = 0
                    else:
                        dp[j] = min(dp[j], 1 + dp[i-1])
            
        return dp[n-1]

    def minCut(self, text: str) -> int:
        n = len(text)
        self.dp = [[-1 for i in range(n)] for j in range(n)]
        res = self.rec(0, n-1, text)
        res = self.memo(0, n-1, text)
        res = self.topdown(text)
        return res

# print( CutsToMakeAllSubsetsPalindrome().minCut("abccbty") )
class MakeExpTrueWays:
    """"""
    def recursive(self, i, j, want: bool, exp: str):
        if i > j:
            return 0
        elif i == j:
            if want:
                return 1 if exp[i] == "T" else 0
            else:
                return 1 if exp[i] == "F" else 0
        res = 0
        for k in range(i+1,j,2):
            lt = self.recursive(i,k-1,True,exp)
            lf = self.recursive(i,k-1,False,exp)
            rt = self.recursive(k+1,j,True,exp)
            rf = self.recursive(k+1,j,False,exp)

            op = exp[k]
            if op == "&":
                if want:
                    res += lt*rt
                else:
                    res += lf*rf + lt*rf + lf*rt   #00 01 10
            elif op == "|":
                if want:
                    res += lf*rt + lt*rf + lt*rt  #01 10 11
                else:
                    res += lf*rf
            elif op == "^":
                if want:
                    res += lt*rf + lf*rt
                else:
                    res += lt*rt + lf*rf
        return res

    def memo(self, i, j, want, exp: str):

        if i > j:
            self.dp[i][j][want] = 0
            return 0
        elif self.dp[i][j][want] != -1:
            return self.dp[i][j][want] #if want else self.dpn[i][j]
        elif i == j:
            res = None
            if want:
                res = 1 if exp[i] == "T" else 0
                self.dp[i][j][want] = res
            else:
                res = 1 if exp[i] == "F" else 0
                self.dp[i][j][want] = res
            return res
        res = 0

        for k in range(i+1,j,2):

            lt = self.memo(i,k-1,1,exp)
            lf = self.memo(i,k-1,0,exp)
            rt = self.memo(k+1,j,1,exp)
            rf = self.memo(k+1,j,0,exp)

            op = exp[k]
            if op == "&":
                if want ==1:
                    res += (lt*rt) #11
                else:
                    res += (lf*rf + lt*rf + lf*rt )  #00 01 10
            elif op == "|":
                if want == 1:
                    res += (lf*rt + lt*rf + lt*rt)  #01 10 11
                else:
                    res += (lf*rf) #00
            elif op == "^":
                if want == 1:
                    res += (lt*rf + lf*rt) #10 01
                else:
                    res += (lt*rt + lf*rf) #11 00
        self.dp[i][j][want] = res
        return res
    
    def solve(self, exp):
        n = len(exp)
        
        self.dp = [[[-1,-1] for i in range(n)] for j in range(n)]
        res = self.memo(0, n-1, 1,exp)
        # res = self.recursive(0,n-1,True,exp)
        return res%1003

print(MakeExpTrueWays().solve("T|F^F&T|F^F^F^T|T&T^T|F^T^F&F^T|T^F"))  #638

            # if self.dp[i][k-1] != -1:
            #     lt = self.dp[i][k-1]
            # else:
            #     lt = self.memo(i,k-1,True,exp)
            # if self.dpn[i][k-1] != -1:
            #     lf = self.dpn[i][k-1]
            # else:
            #     lf = self.memo(i,k-1,False,exp)
            # if self.dp[k+1][j] != -1:
            #     rt = self.dp[k+1][j]
            # else:
            #     rt = self.memo(k+1,j,True,exp)
            # if self.dpn[k+1][j] != -1:
            #     rf = self.dpn[k+1][j]
            # else:
            #     rf = self.memo(k+1,j,False,exp)