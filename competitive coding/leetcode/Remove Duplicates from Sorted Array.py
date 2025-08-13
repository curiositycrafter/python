class Solution(object):
    def removeDuplicates(self, nums):
        a=[]
        for x in nums:
            if x not in a:
                a.append(x)
        nums[:]=a
        return len(nums)
        
