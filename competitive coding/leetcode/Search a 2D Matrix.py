class Solution(object):
    def searchMatrix(self, matrix, target):
        for i in matrix:
            if target in i:
                return target in i
        return False
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        
