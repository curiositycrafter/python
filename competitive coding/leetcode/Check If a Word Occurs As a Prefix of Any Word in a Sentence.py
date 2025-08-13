class Solution(object):
    def isPrefixOfWord(self, sentence, searchWord):
        t=0
        for i in sentence.split():
            t+=1
            if i.startswith(searchWord):
                return t
        return -1
        """
        :type sentence: str
        :type searchWord: str
        :rtype: int
        """
        
