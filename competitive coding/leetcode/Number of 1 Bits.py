class Solution(object):
    def hammingWeight(self, n):
        return((str(bin(n).strip('0b'))).count('1'))
        """
        :type n: int
        :rtype: int
        """
        
