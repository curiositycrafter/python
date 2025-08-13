class MinStack(object):

    def __init__(self):
        self.s=[]
        

    def push(self, val):
        self.s.append(val)
        """
        :type val: int
        :rtype: None
        """
        

    def pop(self):
        self.s.pop()
        """
        :rtype: None
        """
        

    def top(self):
        return self.s[-1]
        """
        :rtype: int
        """
        

    def getMin(self):
        return min(self.s)
        """
        :rtype: int
        """
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
