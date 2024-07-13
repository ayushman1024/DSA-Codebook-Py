class loop_linkedlist:
    #Function to find first node if the linked list has a loop.
    # https://www.geeksforgeeks.org/problems/find-the-first-node-of-loop-in-linked-list--170645/1
    def findFirstNode(self, head):
        #code here
        slow, fast = head, head
        
        # detect loop using two speed pointer approach
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        # if no loop exists
        if slow != fast:
            return -1
        
        # if loop point is at k distanct from head, then fast pointer is k distant behind loop point 
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
            
            if slow == fast:
                break
        return slow.val #slow.data