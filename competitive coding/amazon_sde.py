ip=input().lower()
lim=1
while(True):
    i,j,f=ip[:lim],ip[lim:],1
    for k in i:
        if(k in j):
            f=0
    if(f):
        print(*(i,"self suff") if len(i)>len(j) else (j,"self suff"))
        break
    lim+=1
    
