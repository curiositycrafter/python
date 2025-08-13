# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        t=[]
        while list1!=None:
            t.append(list1.val)
            list1=list1.next
        #t.append(list1.val)
        while list2!=None:
            t.append(list2.val)
            list2=list2.next
        #t.append(list1.val)
        t.sort()
        s=None
        for value in range(len(t), 0, -1):
            s = ListNode(t[value-1], s)
            
        return(s)
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        
