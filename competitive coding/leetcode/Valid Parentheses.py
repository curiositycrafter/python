class Solution(object):
    def isValid(self, s):
        t=[]
        top=-1
        for i in s:
            if i in '({[':
                t.append(str(i))
                top+=1
            if i in '}])':
                if len(t)==0:
                    return False
                if (t[top]=='(' and i==')' )or (t[top]=='[' and i==']' )or(t[top]=='{' and i=='}'):
                    t.pop()
                    top-=1
                else:
                    return False
        return len(t)==0
        """
        :type s: str
        :rtype: bool
        """
        
