x = 8
print((x+1)%9)
def lengthOfLongestSubstring( s: str) -> int:
    chars = [None ] *128

    left = 0
    right = 0
    res = 0
    while right < len(s):
        r = s[right]
        index = chars[ord(r)]
        if index and index < right and index >= left:
            left = index + 1
        # chars[ord(r)] = 0 if not index else index + 1
        chars[ord(r)] = right
        res = max(res, right - left + 1)
        right += 1
    return res


lengthOfLongestSubstring( "abcabcbb" )

class Solution:
    # @param A : list of integers
    # @return an integer
    def solve(self, A):
        n = len(A)
        o_sum = 0
        e_sum = 0
        for index in range(0, n, 2):
            o_sum += A[index]
        for index in range(1, n, 2):
            e_sum += A[index]

        left_odd = 0
        left_even = 0
        output_count = 0

        for i in range(n):
            if (i + 1) % 2 == 0:
                right_odd = o_sum - left_odd
                right_even = e_sum - left_even - A[i]

                odd = left_odd + right_even
                even = right_odd + left_even

                left_even += A[i]
                if even == odd:
                    output_count += 1

            else:
                right_odd = o_sum - left_odd - A[i]
                right_even = e_sum - left_even

                even = left_even + right_odd
                odd = right_even + left_odd

                left_odd += A[i]
                if even == odd:
                    output_count += 1
        return output_count
def prefix_sum(A):
    prefix_sum_arr = [0]*len(A)
    sum_ = 0
    for index in range(len(A)):
        sum_ += A[index]
        prefix_sum_arr[index] = sum_
    return prefix_sum_arr

p_sum = prefix_sum([4,2,0,2,-1,3])
# print(p_sum[3] - p_sum[0])
# print(1^1^1)

sol = Solution()
# print(sol.solve([1,2,3,7,1,2,3]))


def transpose(A):
    row_size = len(A)
    for r in range(1, row_size):
        c = 0
        while c < r:
            t = A[r][c]
            A[r][c] = A[c][r]
            A[c][r] = t
            c += 1
    return A

mat = [[1,2,3],[4,5,6],[7,8,9]]
# print(transpose(mat))
