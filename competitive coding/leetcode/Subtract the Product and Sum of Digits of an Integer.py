class Solution(object):
    def subtractProductAndSum(self, n):
        t=[int(x) for x in str(n)]
        p=1
        for a in t:
            p*=a
        return(p-sum(t))
        """
        :type n: int
        :rtype: int
        """
        
