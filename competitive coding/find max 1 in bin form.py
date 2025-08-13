nos=int(input())
n=list(map(int,input().split()))
for j in range(nos):
    m=n[0]    
    for i in n:
        if bin(i).count('1')>bin(m).count('1'):
            m=i
    print(m)
    n.remove(m)
    
