
class Node:
    """ Node of a trie.
    char: character of the node
    child: list of 26 children, each representing a letter of the alphabet
    is_end: boolean to indicate if the node is the end of any word
    """
    def __init__(self, char=None):
        self.char = char
        self.child = [None]*26
        self.is_end = False

def create_trie(words):
    if type(words) is not list:
        words = words.split(" ")
    trie = Node()
    for word in words:
        temp = trie
        print(word)
        for c in word:
            index = ord(c) - ord("A")
            print(c," ",index)
            if not temp.child[index]:
                temp.child[index] = Node(c)
            temp = temp.child[index]
        temp.is_end = True
    return trie

trie = create_trie("ABC ALO ABXC CLO")
print(trie.child[0].child[11].child)