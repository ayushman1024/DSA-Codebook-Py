import bisect
from typing import List
class Lcs:
    """Longest Common Subsequence
    https://leetcode.com/problems/longest-common-subsequence/
    LCS between two strings is the longest subsequence that is common to both the strings.
    Problem may ask to print the subsequence or length of subsequence or just the length of longest common subsequence.

    """
    def longestCommonSubsequence(self, text1: str, text2: str, return_dp=False) -> int:
        n1, n2 = len(text1), len(text2)
        dp = [ [0 for i in range(n2+1)] for j in range(n1+1) ]

        for i1 in range(1, n1+1):
            for i2 in range(1, n2+1):
                if text1[i1-1] == text2[i2-1]:
                    dp[i1][i2] = 1 + dp[i1-1][i2-1]
                else:
                    dp[i1][i2] = max( dp[i1][i2-1], dp[i1-1][i2] )
        
        if not return_dp:
            return dp[n1][n2]
        else:
            return dp
    
    def printSubsequenceLCS(self, dp, text1, text2):
        """Print the subsequence of LCS.
        1. Start from the bottom-right cell of the matrix and move in the direction of equal cell among top and left.
        2. Else move diagonally up and left and print the character.
        3. Reverse the printed characters to get the subsequence.
        4. Time complexity is O(n+m) and space complexity is O(n+m) where n and m are lengths of input strings.
        """
        out = []
        n1, n2 = len(text1), len(text2)
        while n1 >= 0 and n2 >= 0:
            if dp[n1][n2] == dp[n1-1][n2]:
                n1 -= 1
            elif dp[n1][n2] == dp[n1][n2-1]:
                n2 -= 1
            else:  # only diagonal move to be printed
                out.append(text2[n2-1])
                n1 -=1
                n2 -=1
        out = out[::-1]
        out_str = "".join(s for s in out)
        print(out_str)

    def isSubsequence(self, s: str, t: str) -> bool:
        """https://leetcode.com/problems/is-subsequence/"""
        # DP is not a space/time optimised solution, instead use iteration

        n1, n2 = len(s), len(t)
        prev = [0 for i in range(n2+1)]

        for i1 in range(1,n1+1):
            dp = [0 for i in range(n2+1)]
            for i2 in range(1,n2+1):
                if s[i1-1] == t[i2-1]:
                    dp[i2] = 1 + prev[i2-1]
                else:
                    dp[i2] = max(dp[i2-1], prev[i2])
            prev = dp
        return n1 == 0 or dp[-1] == n1

    def longestCommonSubstring(self, nums1, nums2) -> int:
        n1, n2 = len(nums1), len(nums2)
        dp = [[0 for i in range(n2+1)] for j in range(n1+1)]

        ans = -1
        for i1 in range(1,n1+1):
            for i2 in range(1,n2+1):
                if nums1[i1-1] == nums2[i2-1]:
                    dp[i1][i2] = 1 + dp[i1-1][i2-1]
                else:
                    dp[i1][i2] = 0   # break the chain, bcoz it's substring
                ans = max(ans,dp[i1][i2])
        return ans
    
    def longestPalindromeSubseq(self, s: str) -> int:

        """ Idea is to find length of longest common subsequence between 
        input string and its reversed string."""
        s1 = s
        s2 = s[::-1]
        n1,n2 = len(s1), len(s2)

        dp = [[0 for  i in range(n2+1)] for j in range(n1+1)]
        for i1 in range(1,n1+1):
            for i2 in range(1,n2+1):
                if s1[i1-1] == s2[i2-1]:
                    dp[i1][i2] = 1 + dp[i1-1][i2-1]
                else:
                    dp[i1][i2] = max(dp[i1-1][i2], dp[i1][i2-1])
        
        # to print palindrome subsequence
        self.printSubsequenceLCS(dp,s1,s2)

        # Minimum nos of addition to make input string Palindrom
        # Or Minimum number of deletion to make input string palindrom
        return n2 - dp[n1][n2]
    
        # Length of Palindrome subsequence
        return dp[n1][n2]
    
    def longestRepeatingSubsequence(self, text: str, return_dp=False) -> int:
        n = len(text)
        dp = [ [0 for i in range(n+1)] for j in range(n+1) ]

        for i1 in range(1,n+1):
            for i2 in range(1,n+1):
                if text[i1-1] == text[i2-1] and i1 != i2:
                    dp[i1][i2] = 1 + dp[i1-1][i2-1]
                else:
                    dp[i1][i2] = max( dp[i1][i2-1], dp[i1-1][i2] )
        
        self.printSubsequenceLCS(dp,text,text)

class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        """ https://leetcode.com/problems/shortest-common-supersequence/description/ """
        n1,n2 = len(str1), len(str2)

        dp = [[0 for i in range(n2+1)] for j in range(n1+1)]

        for i1 in range(1,n1+1):
            for i2 in range(1,n2+1):

                if str1[i1-1] == str2[i2-1]:
                    dp[i1][i2] = 1 + dp[i1-1][i2-1]
                else:
                    dp[i1][i2] = max(dp[i1-1][i2], dp[i1][i2-1])
        
        #size of supersequence
        subseq_size = dp[n1][n2]
        super_sequence_size = n1 + n2 - subseq_size

        # building supersequence
        i1,i2 = n1,n2
        out = ""
        while i1 > 0 and i2 > 0:
            if dp[i1][i2] == dp[i1-1][i2] :
                out += str1[i1-1]
                i1 -= 1
            elif dp[i1][i2] == dp[i1][i2-1]:
                out += str2[i2-1]
                i2 -= 1
            else:
                out += str1[i1-1]
                i1 -= 1
                i2 -= 1
        res = out[::-1]
        if i1 > 0:
            res = str1[:i1] + res
        if i2 > 0:
            res = str2[:i2] + res
        return res
    
class MinOpsToMakeCommonString:
    """https://leetcode.com/problems/delete-operation-for-two-strings/description/
    
    Minimum addition and substraction ops to transform "word1" to "word2" 
    1. Idea is to think think problem to find length of longest common subsequence.
    2. uncommon in word1 will be deleted.
    3. uncommon in word2 will be added.
    result = n1 - common + n2 - common
    """

    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0 for i in range(n2+1)] for j in range(n1+1)]

        for i1 in range(1,n1+1):
            for i2 in range(1,n2+1):
                if word1[i1-1] == word2[i2-1]:
                    dp[i1][i2] = 1 + dp[i1-1][i2-1]
                else:
                    dp[i1][i2] = max(dp[i1-1][i2], dp[i1][i2-1])
        
        common = dp[n1][n2]
        return n1 - common + n2 - common

lcs = Lcs()

dp = lcs.longestCommonSubsequence("abcde","ace", return_dp=True)
lcs.printSubsequenceLCS(dp,"abcde","ace")

dp = lcs.longestCommonSubsequence("agbcba","abcbga", return_dp=True)
lcs.printSubsequenceLCS(dp,"agbcba","abcbga")

lcs.longestRepeatingSubsequence("AABEBCDD")

def longestCommonSubstring(nums1, nums2):
    """https://leetcode.com/problems/maximum-length-of-repeated-subarray/
    Given two integer arrays nums1 and nums2, return the maximum length of a subarray that appears in both arrays.
    Time complexity is O(n*m) and space complexity is O(n*m) where n and m are lengths of input arrays.
    Space complexity can be reduced to O(min(n,m)) by using only two rows of dp matrix."""
    n1, n2 = len(nums1), len(nums2)
    dp = [[0 for i in range(n2+1)] for j in range(n1+1)]

    ans = -1
    for i1 in range(1,n1+1):
        for i2 in range(1,n2+1):
            if nums1[i1-1] == nums2[i2-1]:
                dp[i1][i2] = 1 + dp[i1-1][i2-1]
            else:
                dp[i1][i2] = 0   # break the chain, bcoz it's substring
            ans = max(ans,dp[i1][i2])
    return ans

class LongestIncreasingSubsequence:
  """https://leetcode.com/problems/longest-increasing-subsequence/"""

  def lengthOfLIS(self, nums: List[int]) -> int:
    """ O(N*logN)   https://www.youtube.com/watch?v=on2hvxBXJH4"""
    lis = []
    for num in nums:
      pos = bisect.bisect_left(lis,num)
      if pos == len(lis):
        lis.append(num)
      else:
        lis[pos] = num
    return len(lis)

  def lengthOfLIS_v2(self, nums: List[int]) -> int:
    """ DP Solution using 1D array O(N*N).
    Also print one of the Longest Inc subsequence"""
    dp = [1]*len(nums) # LIS ending at ith index
    prev = [None]*len(nums) # keeps track of previous index for every lis ending at i
    res = 1
    for i in range(1,len(nums)):
      for p in range(0,i):
        if nums[p] < nums[i]:
          if dp[p] + 1 > dp[i]:
            dp[i] = 1 + dp[p]
            prev[i] = p

    lis_l, lis_i = 0,0
    for i in range(len(nums)):
      if dp[i] > lis_l:
        lis_l = dp[i]
        lis_i = i
    
    lis = []
    while lis_i is not None:
      lis.append(nums[lis_i])
      lis_i = prev[lis_i]
    lis = lis[::-1]
    print(lis)
    return lis_l
  
  def lengthOfLIS_v1(self, nums: List[int]) -> int:
    """ DP Solution using 1D array O(N*N)."""
    dp = [1]*len(nums) # LIS ending at ith index
    res = 1
    for i in range(1,len(nums)):
      for p in range(0,i):
        if nums[p] < nums[i]:
          dp[i] = max(dp[i],1 + dp[p])
      res = max(res,dp[i])
    return res

  def lengthOfLIS_rec2dp(self, nums: List[int]) -> int:
    """Converted below recursion code to tabulation"""
    n = len(nums)
    dp = [[0]*(n+1) for _ in range(n+1)]

    for i in range(n-1,-1,-1):
      for p in range(i-1,-2,-1):
        if nums[p] < nums[i] or p == -1:
          dp[i][p+1] = max(1 + dp[i+1][i+1], dp[i+1][p+1])
        else:
          dp[i][p+1] = dp[i+1][p+1]
    return dp[0][0]

  def lengthOfLIS_rec(self, nums: List[int]) -> int:
    n = len(nums)

    def rec(i, prev):
      if i == n:
        return 0

      if prev < 0 or nums[prev] < nums[i]: #continue with prev OR new subseq start
        res = max(1 + rec(i+1,i), rec(i+1 , prev))
      else:
        res = rec(i+1,prev)
      return res
    
    return rec(0,-1)

"""
Other variations of LCS are:
1. Longest Common Substring
2. Longest Palindromic Subsequence
3. Longest Repeating Subsequence
4. Shortest Common Supersequence
5. Minimum Operations to Make a String Palindrome
6. Minimum Deletion/Insertion/Addition to Make a String Palindrome
7. Minimum Deletion/Insertion/Addition to Make Two Strings Anagram
8. Longest Increasing Subsequence
9. Longest Bitonic Subsequence
10. Longest Alternating Subsequence
11. Longest Subsequence with Maximum Sum
"""