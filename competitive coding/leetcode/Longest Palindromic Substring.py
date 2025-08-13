class Solution(object):
    def longestPalindrome(self, s):
        m=''
        for i in range(len(s)):
            for j in range(len(s),i,-1):
                p=s[i:j]
                if p==p[::-1] and len(p)>len(m):
                    m=p
        return m
