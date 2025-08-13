import random as r
while True:
    i=input('Enter rock/paper/scissors(r/p/s is also ok) and defeat the computer:')
    k=r.randint(0,2)
    l=['r','p','s']
    if i.lower() in ['r','p','s','rock','paper','scissor']:
        if ((i.lower()=='rock' or i.lower()=='r') and l[k]=='s') or ((i.lower()=='paper' or i.lower()=='p') and l[k]=='r') or((i.lower()=='scissor' or i.lower()=='s') and l[k]=='p'):
            print('You\'ve won!!!')
        else:
            print('You loose')
    else:
        print('Enter a valid option')
        continue
    if input('Would you like to go again').lower() not in ['y','yes','yea','yeah','s','sure','why not']:
        print('Thanks for playing!!!!')
        break
