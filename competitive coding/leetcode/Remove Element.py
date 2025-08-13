class Solution(object):
    def removeElement(self, nums, val):
        l=[]
        for x in nums:
            if x != val:
                l.append(x)
        nums[:]=l
        return len(nums)
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        
