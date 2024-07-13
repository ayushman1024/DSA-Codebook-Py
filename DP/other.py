class LongestPalindrome:
    def longestPalindromeLength(self, s, i, j):
        if i>j:
            return 0
        if i == j:
            return 1
        if s[i] == s[j]:
            return 2 + self.pal(s,i+1,j-1)
        else:
            return max(self.pal(s,i+1,j), self.pal(s,i+1,j))
    
    def longestPalindrome(self, s):
        """https://leetcode.com/problems/longest-palindromic-substring/description/
        Longest palindromic in string s using DP"""
        n = len(s)
        dp = [[False for i in range(n)] for j in range(n)]

        ans = ""
        for leng in range(1,n+1):
            for i in range(0,n-leng):
                j = i + leng - 1
                if s[i] == s[j]:
                    if leng <=2 or dp[i+1][j-1]:
                        dp[i][j] = True
                        if len(ans) < leng:
                            ans = s[i:j+1]
        return ans
    
    def longestPalindromeUnoptimised(self, s: str) -> str:
        n = len(s)

        def ispal(t,i,j):
            while i<= j:
                if t[i] != t[j]:
                    return False
                i += 1
                j -= 1
            return True
        
        dp = [[0 for j in range(n)]for  i in range(n)]
        ans = (0,0)
        for j in range(n):  
            maxs = -1
            max_i = j
            for  i in range(j,-1,-1):
                if dp[i][j] > 0 or ispal(s,i,j):
                    maxs = max(maxs, j-i)
                    max_i = i
            dp[i][j] = maxs
            if ans[1] - ans[0] <  j - max_i:
                ans = (max_i,j)
        return s[ans[0]:ans[1]+1]

print(LongestPalindrome().longestPalindrome("abcbefebk"))


class FrogJump:
    """https://leetcode.com/problems/frog-jump/description/"""
    def canCross(self, stones) -> bool:

        n = len(stones)
        exists = dict()

        for i,pos in enumerate(stones):
            exists[pos] = i
        
        dp = [[False]*n for _ in range(n)]
        dp[0][0] = True

        #dp[pos][k] if pos is reachable with k jump from pos-k stone if it existed

        for posi in range(n):
            for jump in range(posi+1): # max of n-1 jump is possible at nth position
                if not dp[posi][jump]: # If this position stone was never reached by any previous stone - ignore
                    continue
                pos = stones[posi]

                # try all new jumps from pos stone (jump, jump-1, jump+1)
                if pos + jump in exists:
                    dp[exists[pos+jump]][jump] = True
                if pos + jump -1 in exists:
                    dp[exists[pos+jump-1]][jump-1] = True
                if pos + jump +1 in exists:
                    dp[exists[pos+jump+1]][jump+1] = True
        
        return any(dp[n-1][i] for i in range(n))
    

class WilcardMatching:
  """https://leetcode.com/problems/wildcard-matching/description/
  
  Cases: 
  s="x", p=""
  s="",  p="***"
  s="aa", p="*"

  
  """
  def isMatch(self, s: str, p: str) -> bool:
    n1, n2 = len(s), len(p)
    if n2 == 0:
      return n1 == n2

    dp = [[False]*(n2+1) for _ in range(n1+1)]
    dp[0][0] = True  # case: s = "" and p = ""

    for si in range(n1+1):
      for pi in range(1,n2+1):
        if p[pi-1] == "*":
          dp[si][pi] = dp[si][pi-1] 
          if si > 0:
            dp[si][pi] = dp[si][pi] or dp[si-1][pi]
        elif si > 0:
          if p[pi-1] == "?":
            dp[si][pi] = dp[si-1][pi-1]
          else:
            dp[si][pi] = s[si-1] == p[pi-1] and dp[si-1][pi-1]
        
    return dp[n1][n2]