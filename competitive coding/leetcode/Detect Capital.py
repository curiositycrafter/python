class Solution(object):
    def detectCapitalUse(self, word):
        return word.isupper() or word.istitle() or word.islower()
        """
        :type word: str
        :rtype: bool
        """
        
