class Solution(object):
    def getRow(self, rowIndex):
        r=[]
        for i in range(rowIndex+1):
            nr=[]
            for j in range(i+1):
                if j==i or j==0:
                    nr.append(1)
                else:
                    nr.append(r[j-1]+r[j])
            r=(nr)
        return r
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        
