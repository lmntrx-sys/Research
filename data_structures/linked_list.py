# create a self referential structure for a 
# linked list

class Node:
    def __init__(self, data):
        """
        Initializes a new instance of the Node class with the provided data.

        Parameters:
            data (Any): The data to be stored in the node.

        Returns:
            None
        """
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        
    def print_list(self):
        """
        Prints the data of each node in the linked list.

        This function iterates through the linked list starting from the head node 
        and prints the data of each node until the end of the list is reached.

        Parameters:
            self (LinkedList): The linked list object.

        Returns:
            None
        """
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next

    def append(self, data):
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
            return self.head
        
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node
        return self.head
    
    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        return self.head
    
    def insert_after_node(self, prev_node, data):
        """
        Inserts a new node after a given previous node in the linked list.

        Args:
            prev_node (Node): The previous node before which the new node will be inserted.
            data (Any): The data to be stored in the new node.

        Returns:
            Node: The head of the linked list after insertion.

        Raises:
            None

        Note:
            - If the given previous node is None, a message is printed and the head of the linked list is returned.
            - The new node is inserted after the given previous node by updating the next pointer of the previous node to point to the new node.
            - The new node's next pointer is set to the original next pointer of the previous node.
        """
        if prev_node is None:
            print("The given previous node must not be null")
            return self.head
        new_node = Node(data)
        new_node.next = prev_node.next
        prev_node.next = new_node
        return self.head
    
    def delete_node(self, key):
        """
        A function to delete a node with a given key from the linked list.

        Parameters:
            key: The key value of the node to be deleted.

        Returns:
            The head of the linked list after deletion.
        """
        temp = self.head
        if temp is not None:
            if temp.data == key:
                self.head = temp.next
                temp = None
                return self.head
        while temp is not None:
            if temp.data == key:
                break
            prev = temp
            temp = temp.next
        if temp == None:
            return self.head
        prev.next = temp.next
        temp = None
        return self.head
    
x= dict({'1': 1, '2': 2, '3': 3})
print(x['1'])

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

      # Create a list which we eill iterate over
      left = 0
      max_len = 0
      char_set = set()
      
      for right in range(len(s)):
        while s[right] in char_set:
          char_set.remove(s[left])
          left += 1

        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

      return max_len