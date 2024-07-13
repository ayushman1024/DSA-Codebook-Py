import math

def rotateL(A, B):
    arr = A + A
    n = len(A)
    for b in B:
        print( arr[b:b+n] )
def rotateR(A, B):
    arr = A + A
    n = len(A)
    for b in B:
        print(arr[n-b:2*n-b])

print(rotateR([1,2,3,4,5],[2]))

"""
1. Make prime bool array of size N+1 (initialised with True)
2. loop i from 2 to sqrt(N)
3. That is for each number you mark it multiples starting from its square(not from nos*2)
4. if prime[i] > 0 : leave i and mark it multiples as False
5. Return "Primes" Array
Time Complexity: O(n*log(log(n)))
Auxiliary Space: O(n)
sqrt(n) *
https://iq.opengenus.org/sieve-of-eratosthenes-analysis/

[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
"""
def sieve_prime(N):
    primes = [True for i in range(N+1)]
    primes[0],primes[1] = False, False
    i = 2
    while i <= math.sqrt(N):
        if primes[i]:
            for j in range(i*i,N+1,i):
                primes[j] = False
        i += 1
    return primes

class MinMax:
    # @param A : list of integers
    # @return an integer
    def get_minmax(self, A, l, r):
        # return tuple(min,max)
        if r-l == 0:
            return ( A[l], A[l] )

        if r-l == 1:
            if A[l] < A[r]:
                return (A[l],A[r])
            return (A[r],A[l])

        m = l + (r-l)//2

        [min_l, max_l] = self.get_minmax(A,l,m)
        [min_r, max_r] = self.get_minmax(A,m+1,r)

        _min = min_l if min_l < min_r else min_r
        _max = max_l if max_l > max_r else max_r

        return (_min, _max)

    def solve(self, A):
        [_min,_max] = self.get_minmax(A,0,len(A)-1)
        return _min + _max

def slide_window(A,k):
    n = len(A)
    sum_ = sum(A[0:k])
    for i in range(1,n-k+1):
        sum_ += A[i+k-1] - A[i-1]
        print(sum_,end="\n")

def anti_diagonal(A):
    N = len(A)
    # output = [[0 for i in range(N)] for j in range(2*N -1)]
    output = []
    for col in range(0,N):
        r,c = 0,col
        i = 0
        out = [ 0 for i in range( N ) ]
        while N > r >= 0 and N > c >= 0:
            out[i] = A[r][c]
            r += 1
            c -= 1
            i += 1
        output.append(out)
    for row in range(1,N):
        r,c = 1,N-1
        i = 0
        out = [0 for i in range(N)]
        while N > r >= 0 and N > c >= 0:
            out[i] = A[r][c]
            r += 1
            c -= 1
            i += 1
        output.append(out)
    return output


def lengthOfLongestSubstring(self, s: str) -> int:
    #https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
    if len(s) <= 1:
        return len(s)
    res = 1
    left = 0
    right = 1

    cset = dict()
    cset[s[0]] = 0

    while left <= right and right < len(s):
        r_char = s[right]
        if r_char in cset:
            #if repeat char, move left to just next of prev occurence, and del in between char from map
            old_index = cset[r_char]
            newleft = old_index + 1

            # remove others from dict cset
            for i in range(left, newleft):
                del cset[s[i]]
            
            cset[r_char] = right
            left = newleft
        else:
            cset[s[right]] = right
            res = max(res, right - left + 1)
        right += 1
    return res

print(anti_diagonal([[1,2,3],[4,5,6],[7,8,9]]))

minmax = MinMax()
print( minmax.solve( [13,77,8,88,43,10,6,66] ) )
rotateL( [1,2,3,4,5,6,7], [2,5,3,0,1] )
print("\n")
rotateR( [1,2,3,4,5,6,7], [2,5,3,0,1] )
print("\n")
print(sieve_prime(10))
print("\n")
slide_window([1,2,3,4,5,6],3)

def merge( intervals):
        n = len(intervals)
        intervals.sort(key = lambda x:x[0])
        out = []
        prev = intervals[0]
        for i in range(1,n):
            if intervals[i][0] <= prev[1]:
                prev[1] = max(prev[1],intervals[i][1])
            else:
                out.append(prev)
                prev = intervals[i]
        if prev:
            out.append(prev)
            
        return out

print(merge([[1,2],[3,4],[6,7]]))