class node:
    def __init__(self,val):
        self.prev=None
        self.val=val
        self.next=None
def printt(n):
    while n.next:
        print(n.val,end='->')
        n=n.next
    print(n.val)
def print2(n):
    while n.prev:
        print(n.val,end='->')
        n=n.prev
    print(n.val)
def ins(head,val,pos,las):
    n=node(val)
    if pos==1:
        n.next=head
        n.prev=las
        head.prev=n
    else:
        no=head
        for i in range(pos-1):
            no=no.next
        n.next=no.next
        n.next.prev=n#this line is something that u may miss
        n.prev=no
        no.next=n        
n1=node(1)
n2=node(2)
n3=node(3)
n4=node(4)
n5=node(5)
n1.prev,n2.prev,n3.prev,n4.prev,n5.prev=n5,n1,n2,n3,n4
n1.next,n2.next,n3.next,n4.next,n5.next=n2,n3,n4,n5,n1
ins(n1,45,3,n5)
printt(n1)
print2(n5)
