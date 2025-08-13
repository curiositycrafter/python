#this is a basic project that lets a user guess a number generated in random
import random as r
while True:
    ui=input("Enter a number for you to guess starting with 0 ")
    if ui.isdigit():
        if int(ui)>0:
            i=r.randint(0,int(ui))#includes upper range too
            tries=0
            ans=[]
            while True:
                q=input("Guess the number:")
                tries+=1
                if q.isdigit():
                    if int(q)>=0 and int(q)<=int(ui):
                        if int(q) in ans:
                            print('Don\'t enter the same number again!!')
                            tries-=1
                        else:
                            ans.append(int(q))
                            if i==int(q):
                                print(f'You have got it right in the {tries} tries')
                                break
                            else:                        
                                print("Try again")
                    else:
                        print('Enter a number within the specified range!!')
                else:
                    print("Please enter a digit")
            print("Hope you enjoyed the game!!")
            if input("Would you like to go again?") not in ['y','s','yep','yes','yea','maybe','sure','why not']:
                break
        else:
            print('Enter a number greater than 0')
    else:
        print('Enter a number')
    
