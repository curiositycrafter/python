class Solution(object):
    def lengthOfLastWord(self, s):
        s=s.strip()
        s=s.split()
        return len(s[-1])
        """
        :type s: str
        :rtype: int
        """
        
