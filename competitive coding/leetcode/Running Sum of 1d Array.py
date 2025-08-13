class Solution(object):
    def runningSum(self, nums):
        s=nums[0]
        for i in range(1,len(nums)):
            nums[i]+=s
            s=nums[i]
        return nums
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
