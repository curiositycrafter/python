class Solution(object):
    def maxProfit(self, prices):
        # a=min(prices)
        # b=prices.index(a)
        # #print(b)
        # return max(prices[b:])-a
        pro=0
        min=prices[0]
        for i in prices:
            min=i if i<min else min
            pro=i-min if i-min>pro else pro
        return pro


        """
        :type prices: List[int]
        :rtype: int
        """
        
