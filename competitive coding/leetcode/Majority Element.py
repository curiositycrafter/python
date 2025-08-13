from collections import Counter
class Solution(object):
    def majorityElement(self, nums):
        c=Counter(nums)
        for i in range(len(nums)):
            if c[nums[i]]>len(nums)/2:
                return nums[i]
        """
        :type nums: List[int]
        :rtype: int
        """
        
