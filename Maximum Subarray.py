class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        ma = nums[0]
        for i in range(1, len(nums)):
            dp[i] = nums[i] + (dp[i-1] if dp[i-1]>0 else 0)
            ma = max(dp[i], ma)
        return ma
