class node:
    def __init__(self,val):
        self.val=val
        self.next=None
def printt(n):
    while n.next:
        print(n.val,end='->')
        n=n.next
    print(n.val)
def ins(head,val,pos):
    n=node(val)
    if pos==1:
        n.next=head
    else:
        no=head
        for i in range(pos-1):
            if no.next is None:
                print('error')
            no=no.next
        n.next=no.next
        no.next=n        
n1=node(1)
n2=node(2)
n3=node(3)
n4=node(4)
n5=node(5)
n1.next,n2.next,n3.next,n4.next=n2,n3,n4,n5
ins(n1,45,3)
printt(n1)
