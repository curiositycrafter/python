class Solution(object):
    def plusOne(self, digits):
        a=list(map(str,digits))
        return [int(x) for x in str(int(''.join(a))+1)]
        
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        
