if input('Do u wanna ans a set of qs? ').lower() not in ['y','yes','s','yea','yep']:
    quit()
s=0
if input('whats is the capital of japan? ').lower()=='tokyo':
    print('Your right!!')
    s+=1
else:
    print('nope')
if input('whats is the capital of fance? ').lower()=='italy':
    print('Your right!!')
    s+=1
else:
    print('nope')
if input('whats is the capital of china? ').lower()=='beijing':
    print('Your right!!')
    s+=1
else:
    print('nope')
if input('whats is the capital of Wales? ').lower()=='sydney':
    print('Your right!!')
    s+=1
else:
    print('nope')
if input('where is spiderman from?  ').lower()=='new york':
    print('Your right!!')
    s+=1
else:
    print('nope')
print('Your score is: ',s,'and thats ',s/5*100,'%')
