class Solution(object):
    def generate(self, numRows):
        l=[]
        r=[]
        for i in range(numRows):
            nr=[]
            for j in range(i+1):
                if j==i or j==0:
                    nr.append(1)
                else:
                    nr.append(r[j-1]+r[j])
            l.append(nr)
            r=(nr)
        return l
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        
