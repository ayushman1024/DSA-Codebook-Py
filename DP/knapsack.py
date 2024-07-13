class Knapsack_01:
    """Given n items and bag of max capacity weight W. You need to maximise the value added by adding items to bag.
    given Value[] and Weight[] arrays for each items"""

    def recursion(self, wt, val, w, n): 
        """ Time Complexity: O(2^N) 
            Auxiliary Space: O(N) """
        if n == 0 or w == 0: #if no item available or 0 bag capacity -> 0 value return
            return 0
        if wt[n-1] <= w:  # if nth item's weigth less than available capacity to fill ?
            value = max( val[n-1] + self.recursion( wt, val,w - wt[n-1], n-1 ), self.recursion( wt, val, w, n-1 ) )  #may or may not pick nth item
        else:
            value = self.recursion( wt, val, w, n-1 )
        return value
    
    def memoization(self, wt, val, w, n):
        """ Time Complexity: O(N * W). As redundant calculations of states are avoided.
            Auxiliary Space: O(N * W) + O(N). """
        if n == 0 or w == 0:
            return 0
        if self.dp[n][w] != -1:
            return self.dp[n][w]
        if wt[n-1] <= w:
            self.dp[n][w] = max( val[n-1] + self.memoization(wt, val, w - wt[n-1], n-1), self.memoization(wt, val, w, n-1) )
        elif wt[n-1] > w:
            self.dp[n][w] = self.memoization(wt, val, w, n-1)
        return self.dp[n][w]
    
    def topdown(self, wt, val, w, n):
        """ Time Complexity: O(N * W). where ‘N’ is the number of elements and ‘W’ is capacity. 
            Auxiliary Space: O(N * W). The use of a 2-D array of size ‘N*W’.
        """
        for ni in range(1,n+1): # v.imp   here 'ni' means ni'th position/ till Ni'th position in array: so we will do -1 in below code for 0 based indexing
            for vac in range(1, w+1):

                if wt[ni-1] <= vac:  #if previous didn't filled bag completely
                    self.dp[ni][vac] = max( val[ni-1] + self.dp[ni-1][vac - wt[ni-1]] ,     # selecting ni'th item
                                            self.dp[ni-1][vac])                             # not selecting ni'th item
                else:
                    self.dp[ni][vac] = self.dp[ni-1][vac]
    
        return self.dp[n][w]

    def knapSack(self,W, wt, val, n):
        
        self.dp = [ [-1 for _ in range(W+1)] for __ in range(n+1) ]

        for _ in range(n+1): # for 0 items, max value will always be 0
            self.dp[_][0] = 0
        for _ in range(W+1): # for 0 available weight in bag, max value will always be 0
            self.dp[0][_] = 0

        # out = self.recursion( wt, val, W, n)
        # out = self.memoization(wt, val, W, n)
        # print(out)
        self.topdown(wt, val, W, n)
        print(self.dp)
        return self.dp[n][W]

drive = Knapsack_01()
print("Basic Knapsack max val problem: ",drive.knapSack(14,[2,5,3,2,1], [3,7,5,1,9],5))

#=========================================================================================

def isSubsetSum ( N, arr, sum) -> bool:
    #https://www.geeksforgeeks.org/problems/subset-sum-problem-1611555638/1
    """Check if there exists any subset in "arr" with sum = sum """
    dp = [[False for _ in range(sum+1)] for __ in range(N+1)] # dp[i][s]'s bool value means if sum=s possible with first i elements of array?
    
    for i in range(N+1):
        for s in range(sum+1):
            if s == 0:
                dp[i][s] = True
            elif i == 0 :
                dp[i][s] = False
            elif s >= arr[i-1]:
                dp[i][s] = dp[i-1][s - arr[i-1]] or dp[i-1][s]  # NOTE: here we are or'ing
            else:
                dp[i][s] = dp[i-1][s]
    return dp[N][sum]


#====================================================================================
class EqualPartitionSum:
    """ If array can be divide into two subset of equal sum?
        It is only possible when -> total sum of arr is even and there exist any subset with half the total array sum
        We can use subset sum problem to solve this question"""

    def isSubsetSum (self, N, arr, sum) -> bool:

        """Check is there exist subset in "arr" with sum = sum """
        # dp[i][s]'s bool value means if sum=s possible with first i elements of array?
        dp = [[False for _ in range(sum+1)] for __ in range(N+1)] 
    
        for i in range(N+1):
            for s in range(sum+1):
                if s == 0:
                    dp[i][s] = True
                elif i == 0 :
                    dp[i][s] = False
                elif s >= arr[i-1]:
                    dp[i][s] = dp[i-1][s - arr[i-1]] or dp[i-1][s]  # NOTE: here we are or'ing
                else:
                    dp[i][s] = dp[i-1][s]
        return dp[N][sum]

    def equalPartition(self, N, arr):
        total = sum(arr)
        return total%2 == 0 and self.isSubsetSum(N, arr, total//2)

def countSubsetWithSum ( N, arr, sum) -> bool:
    """Check if there exist subset in "arr" with summation = sum """

    # dp[i][s]'s count value is number of subsets possible with sum=s with first i elements of array.
    dp = [[0 for _ in range(sum+1)] for __ in range(N+1)] 

    for _ in range(N+1):
        dp[_][0] = 1        # there can always be 1 null subset with nay number of element to create target sum=0
    for i in range(1,N+1):
        for s in range(1,sum+1):
            if s >= arr[i-1]:
                dp[i][s] = dp[i-1][s] + dp[i-1][s - arr[i-1]]       # NOTE: here we are adding
            else:
                dp[i][s] = dp[i-1][s]

    return dp[N][sum]
print("Subset count with given sum: ",countSubsetWithSum(4,[1,1,2,3],5))

class minDiffTwoSubset:
    """Minimum sum partition (for non-negative numbers)
    Divide array in two subset so that their sum diff is mimimum.
    
    Approach: We use last row and find for all Si if S - Si is also possible. """
    def subsetsum(self,n,s,nums):
        dp = [ [False for _ in range(s+1)] for k in range(n+1) ]

        for i in range(n+1):
            dp[i][0] = True
        
        for ni in range(1,n+1):
            for si in range(1,s+1):
                if nums[ni-1] <= si:
                    dp[ni][si] = dp[ni-1][si] or dp[ni-1][si - nums[ni-1]]
                else:
                    dp[ni][si] = dp[ni-1][si]
        return dp[-1]  # return last row of dp array

    def minDifference(self, arr, n):
        s = sum(arr)
        dp_ = self.subsetsum(n, s, arr)
        m = s//2
        min_diff = float("inf")

        for i in range(0,m+1):  #loop till middle
            if dp_[i] and dp_[s-i]: # i and s-i can be equal, which means two equal sum subset
                min_diff = min( min_diff, s-2*i )   #abs((s-i) - i) -> s - 2*i
        return min_diff
    
class countSubsetWithGivenDiff:
    """ Count all possible partitions when difference between two subsets is given diff in input.
      Problem narrows down to finding count of all subset whose sum is calc_sum = (arr_sum + target_diff)//2 
      And this has been already solved previously."""
    def solve(self, arr, diff):
        n = len(arr)
        s = sum(arr)
        calc_sum = (s + diff)//2
        dp = [ [0 for _ in range(calc_sum+1)] for __ in range(n+1)]
        for _ in range(n+1):
            dp[_][0] = 1
        for ni in range(n+1):  #
            for si in range(calc_sum+1):
                if si == 0:
                    dp[ni][si] = 1
                elif arr[ni-1]<= si: #if prev didn't overflowed
                    dp[ni][si] = dp[ni-1][si] + dp[ni-1][si - arr[ni-1]] 
                else:
                    dp[ni][si] = dp[ni-1][si]
        return dp[n][calc_sum]


print("count subset with diff: ",countSubsetWithGivenDiff().solve([1,1,2,3],1))

#===========================================
""" 
Target Sum
https://leetcode.com/problems/target-sum/

You are given an integer array nums and an integer target.

You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.

Example:
    [1,1,2,3]   s=1
    +1 -1 -2 +3
    -1 +1 -2 +3
    +1 +1 +2 -3

    p + n = diff
    p - n = sum
    2p = diff + sum
    p = (diff + sum)//2 

Approach:
Array can be divided into two group 'n' and 'p' where we add '-' and '+' symbol to their elements respectively.
And by above calculation we need to find number of ways to make subsets with summation = 'p'
0's are handled separately
Note:

1. If diff + sum is not an even number return 0 as final answer

0's must be handled separetly.
1. create another array with non-zeros, find solution as taught in video.
2. count zeros in original input and final answer will be ANS * pow(2,zero_count)
 """
class TargetSum:
    def findTargetSumWays(self, nums, target) -> int:
        arr = [elem for elem in nums if elem > 0]
        zeros = 0

        for elem in nums:
            if elem == 0:
                zeros += 1

        n = len(arr)
        s = sum(arr)
        target = abs(target)

        if (s + target)%2 != 0:
            return 0

        p = (s + target)//2
        dp = [[0 for i in range(p+1)] for j in range(n+1)]

        for i in range(n+1):
            dp[i][0] = 1

        for ni in range(1,n+1):
            for si in range(1,p+1):
                if arr[ni-1] <= si:
                    dp[ni][si] = dp[ni-1][si] + dp[ni-1][si - arr[ni-1]]
                else:
                    dp[ni][si] = dp[ni-1][si]

        return pow(2,zeros)*dp[n][p]
print("target sum: ",TargetSum().findTargetSumWays([1,0,0,0,0,0,0,0,0],2))


class RodCut:
    def solve(self, l, r, cuts):
        val = float("inf")
        if (l,r) in self.memo:
            return self.memo[(l,r)]
        if len(cuts) == 0 or r-l == 1:
            self.memo[(l,r)] = 0
            return 0
        if len(cuts) == 1:
            self.memo[(l,r)] = r-l
            return r-l
        for cut in cuts:
            new_cuts1 = [c for c in cuts if l< c < cut]
            new_cuts2 = [c for c in cuts if cut < c < r ]

            cut_val = self.solve(l, cut, new_cuts1) + self.solve(cut, r, new_cuts2)
            if cut_val < val:
                val = cut_val
        ans = r-l+val
        self.memo[(l,r)] = ans
        return ans

    def minCost(self, n: int, cuts) -> int:
        self.memo = dict()
        return self.solve(0,n,cuts)
    
cuts = [13,25,16,20,26,5,27,8,23,14,6,15,21,24,29,1,19,9,3]
print("running")
print("Rod cut: ",RodCut().minCost(30,cuts))