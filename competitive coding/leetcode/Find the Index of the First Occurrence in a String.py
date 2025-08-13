class Solution(object):
    def strStr(self, haystack, needle):
        try:
            return haystack.index(needle)
        except:
            return -1
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        
