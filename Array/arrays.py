
from typing import List


def containerWithMostWater(height) -> int:
    """https://leetcode.com/problems/container-with-most-water/"""
    n = len(height)
    l,r = 0,n-1
    res = -1
    while l <r:
        water = (r-l)*min(height[l], height[r])
        res= max(res,water)
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return res

print(containerWithMostWater([1,8,6,2,5,4,8,3,7]))

class Solution:
    """https://leetcode.com/problems/two-sum/solutions/4605441/beats-99-other-solutions-linear-time-and-space-complexity-6-line-python-code"""
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mapping = dict()
        for i,val in enumerate(nums):
            other = target - val
            if other in mapping:
                return [i, mapping[other]]
            mapping[val] = i

class KadanesAlgorithm:
    """To find the maximum subarray sum possible in the given array.
    https://leetcode.com/problems/maximum-subarray/
    """
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        max_till = min(nums)
        sum_ = 0

        for num in nums:
            sum_ += num
            max_till = max(sum_, max_till)
            if sum_ <= 0:
                sum_ = 0
        return max_till

class LongestWellPerformingInterval:
    def longestWPI(self, hours: List[int]) -> int:
        """https://leetcode.com/problems/longest-well-performing-interval/solutions/
        Problem statement: Given a list of hours, find the longest interval where sum of hours is greater than 8.
        Solution: Convert hours to another array where 1 represents hours > 8 and -1 represents hours <= 8.
        Then find the longest subarray with sum > 0.
        And to find the longest subarray with sum > 0, use a hashmap to store the sum and its index.
        For every prefix sum, store the index in hashmap. If prefix sum-1 is already in hashmap, then the subarray from that index to current index is a valid subarray.
        Update answer with the length of this subarray after comparing with previous answer."""

        binary = [-1 if v <= 8 else 1 for v in hours]
        summap = dict()
        prev = 0
        ans = 0
        for i,h in enumerate(binary):
            prev = prev+h
            if prev > 0: # only when longest interval starts from first index itself.
                ans = i + 1
            else:
                if prev not in summap:
                    summap[prev] = i
                if prev-1 in summap:
                    ans = max(ans, i - summap[prev-1])
        return ans