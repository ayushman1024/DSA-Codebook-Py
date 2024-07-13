class Knapsack_unbounded:
    
    def recursion(self, wt, val, w, n):
        """ Time Complexity: O(2^N) 
            Auxiliary Space: O(N) """
        if n == 0 or w == 0:
            return 0
        if wt[n-1] < w:
            value = max( val[n-1] + self.rec( wt, val,w - wt[n-1], n ), self.rec( wt, val, w, n-1 ) )
        else:
            value = self.rec( wt, val, w, n-1 )
        return value
    
    def memoization(self, wt, val, w, n):
        """ Time Complexity: O(N * W). As redundant calculations of states are avoided.
            Auxiliary Space: O(N * W) + O(N). """
        if n== 0 or w == 0:
            return 0
        if self.dp[n][w] != -1:
            return self.dp[n][w]
        if wt[n-1] <= w:
            self.dp[n][w] = max( val[n-1] + self.memoization(wt, val, w - wt[n-1], n), self.memoization(wt, val, w, n-1) )
        elif wt[n-1] > w:
            self.dp[n][w] = self.memoization(wt, val, w, n-1)
        return self.dp[n][w]
    
    def topdown(self, wt, val, w, n):
        """ Time Complexity: O(N * W). where ‘N’ is the number of elements and ‘W’ is capacity. 
            Auxiliary Space: O(N * W). The use of a 2-D array of size ‘N*W’.
        """
        for ni in range(1,n+1): # v.imp   here 'ni' means ni'th position/ till Ni'th position in array: so we will do -1 in below code for 0 based indexing
            for wi in range(1,w+1):

                if wt[ni-1] <= wi:  #if previous didn't filled bag completely
                    self.dp[ni][wi] = max( val[ni-1] + self.dp[ni][wi - wt[ni-1]] , self.dp[ni-1][wi])
                else:
                    self.dp[ni][wi] = self.dp[ni-1][wi]
    
        return self.dp[n][w]

    def knapSack(self,W, wt, val, n):
        
        self.dp = [ [-1 for _ in range(W+1)] for __ in range(n+1) ]

        for _ in range(n+1):
            self.dp[_][0] = 0
        for _ in range(W+1):
            self.dp[0][_] = 0

        # out = self.recursion( wt, val, W, n)
        # out = self.memoization(wt, val, W, n)
        self.topdown(wt, val, W, n)
        return self.dp[n][W]

drive = Knapsack_unbounded()
print("Basic Unbounded Knapsack max val problem: ",drive.knapSack(4,[4,5,1], [1,2,3],3))

class RodCutting:
    """https://www.geeksforgeeks.org/problems/rod-cutting0840/1 Maximise the value of rod by cutting it into pieces and selling them."""
    def cutRod(self, price, n):
        dp = [[0 for l in range(n+1)] for ind in range(n+1)]
        
        for ind in range(n+1):
            for ln in range(n+1):
                if ln == 0 or ind == 0:
                    dp[ind][ln] = 0
                elif ln >= ind:
                    dp[ind][ln] = max(price[ind-1] + dp[ind][ln - ind], dp[ind-1][ln])
                else:
                    dp[ind][ln] = dp[ind-1][ln]
        return dp[n][n]
    
    def cutRodOptimised(self, price, n):
        dp = [0]*(n+1)
        
        for ind in range(n+1):
            for ln in range(n+1):
                if ln == 0 or ind == 0:
                    dp[ln] = 0
                elif ln >= ind:
                    dp[ln] = max(price[ind-1] + dp[ln - ind], dp[ln])
                else:
                    dp[ln] = dp[ln]
        return dp[n]

    def cutRodOptimised2(self,price, n):
        dp = [0]*(n+1)
        
        for ind in range(1,n+1):  # iterate to possible length of rod
            for cut in range(1,ind+1): # iterate to all possible size cut
                dp[ind] = max(price[cut-1] + dp[ind - cut], dp[ind])

        return dp[n]

RodCutting().cutRod([1, 5, 8, 9, 10, 17, 17, 20],8)
RodCutting().cutRodOptimised([1, 5, 8, 9, 10, 17, 17, 20],8)
RodCutting().cutRodOptimised2([1, 5, 8, 9, 10, 17, 17, 20],8)

class CoinChange:

    def coinChange_1(self, coins, amount: int) -> int:
        """https://leetcode.com/problems/coin-change/description/
        Minimum number of coin required to get an amount.
        Unlimited supply coin
        It is unbounded knapsack problem"""
        n = len(coins)

        dp = [ [float("inf") for amt in range(amount+1)] for coin_index in range(n+1)]

        for ci in range(n + 1): # base condition: if amount = 0 then nos of coin required=0
            dp[ci][0] = 0  

        for ci in range(1,n+1):
            for amt in range(1,amount+1):
                if coins[ci-1] <= amt:
                    dp[ci][amt] = min( 1 + dp[ci][amt - coins[ci-1]], dp[ci-1][amt] )
                else:
                    dp[ci][amt] = dp[ci-1][amt]

        return -1 if dp[n][amount] == float("inf") else dp[n][amount]  # inf if no possible combination possible for amt
                
    def coinChange_2(self, amount: int, coins) -> int:
        """https://leetcode.com/problems/coin-change-ii/
        Unlimited supply coin
        Number of ways to achieve required amount"""
        n = len(coins)

        dp = [ [0 for amt in range(amount+1)] for coin_index in range(n+1)]
        for ci in range(n + 1):  # base condition: if amount = 0 then nos of ways will be 1 i.e with no coin
            dp[ci][0] = 1
        for ci in range(1,n+1):
            for amt in range(1,amount+1):
                if coins[ci-1] <= amt:
                    dp[ci][amt] = dp[ci-1][amt] + dp[ci][amt - coins[ci-1]]  # sum of ways by: picking/not picking ci'th coin
                else:
                    dp[ci][amt] = dp[ci-1][amt]
        return dp[n][amount]
    

CoinChange().coinChange_1([2,5,10],15)