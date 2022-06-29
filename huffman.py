# A Huffman Tree Node
from collections import Counter

class HuffmanTree:
    def __init__(self, chars):
        self.chars = chars
        self.bit_count = {}
        self.freq = []
        self.input = []

    class node:
        def __init__(self, freq, symbol, left=None, right=None):
            # frequency of symbol
            self.freq = freq

            # symbol name (character)
            self.symbol = symbol

            # node left of current node
            self.left = left

            # node right of current node
            self.right = right

            # tree direction (0/1)
            self.huff = ''

    # utility function to print huffman
    # codes for all symbols in the newly
    # created Huffman tree
    def printNodes(self, node, val=''):
        # huffman code for current node
        newVal = val + str(node.huff)
        # if node is not an edge node
        # then traverse inside it
        if(node.left):
            c = self.printNodes(node.left, newVal)
            if type(c) == int:
                i += c
        if(node.right):
            c = self.printNodes(node.right, newVal)
            if type(c) == int:
                i += c

            # if node is edge node then
            # display its huffman code
        if(not node.left and not node.right):
            print(f"{node.symbol} -> {newVal}")
            self.bit_count[node.symbol] = newVal
            


    # characters for huffman tree
    def main(self):
        # chars = ['10', '34', '10', '8', '10', '10', '127', '43', '6', '34', '10', '5', '34', '8', '8']
        self.input = [key for key, value in Counter(self.chars).most_common()]
        self.input.reverse()
        # print('Bits need befor huffman coding: ', len(self.chars) * 8)

        # frequency of characters

        for i in self.input:
            self.freq.append(self.chars.count(i))

        # list containing unused nodes
        nodes = []

        # converting characters and frequencies
        # into huffman tree nodes
        for x in range(len(self.input)):
            nodes.append(self.node(self.freq[x], self.input[x]))

        while len(nodes) > 1:
            nodes = sorted(nodes, key=lambda x: x.freq)
            # pick 2 smallest nodes
            left = nodes[0]
            right = nodes[1]

            # assign directional value to these nodes
            left.huff = 0
            right.huff = 1

            # combine the 2 smallest nodes to create
            # new node as their parent
            newNode = self.node(left.freq+right.freq, left.symbol+right.symbol, left, right)

            # remove the 2 nodes and add their
            # parent as new node among others
            nodes.remove(left)
            nodes.remove(right)
            nodes.append(newNode)

        # Huffman Tree is ready!
        # self.printNodes(nodes[0])
        # i = 0
        # for item in self.bit_count:
        #     i += self.bit_count[item]
        # print('Bits need after huffman coding: ',i)
        return self.bit_count
