n=int(input())
r=[]
for i in range(n):
    nr=[]
    for j in range(i+1):
        if j==0 or j==i:
            nr.append(1)
        else:
            nr.append(r[j-1]+r[j])
    print(*nr)#unpacks the values one by one with space
    r=nr
            
