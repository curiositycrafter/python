class Solution(object):
    def romanToInt(self, s):
        a = [('M',1000), ('CM',900), ('D',500), ('CD',400), ('C',100),
             ('XC',90), ('L',50), ('XL',40), ('X',10),
             ('IX',9), ('V',5), ('IV',4), ('I',1)]
        summ=0
        while len(s)>0:
            for x,y in a:
                if s.startswith(x):
                    summ+=y
                    s=s[len(x):]
                    break
        return summ
