## Linked List / Pointer Representation:

# Vertex in a directed graph; intex,outex: in/out vertex edges
class Node:
    def __init__(self, intex, outex, weight):
        self.weight = weight
        self.intex = intex
        self.outex = outex

# Directed Graph
class Graph:
    def __init__():
        # TODO




def old_Q(state, action):


def new_Q(state, action):
    return old_Q()


## Hash Table (Dictionary) Representation:
# Creates a simple two-way n-ary tree
class Graph:
    def __init__(self, depth, actions):
        self.depth = depth
        self.actions = actions
        self.graph = {}

        self.__addNode('root', 'state_00', 0.0)
        self.__populate()

    def __addNode(self, sup, sub, w):  # superior, subordinate, weight
        if not sup in self.graph.keys():
            self.graph[sup] = []
        self.graph[sup].append([sub, w])
        self.graph[sub] = []
        self.graph[sub].append([sup, w])

    def __populate(self):
        for d in range(self.depth):
            for a in range(len(self.actions)):
                self.__addNode('state_0' + str(d),
                             'state_' + str(d+1) + chr(ord('A') + a),
                             self.actions[a])

graph = Graph(1, [2,3])

graph.graph

for node in graph.graph:
    print(node)


graph = {}
'root' in graph.keys()

if graph['root']:
    graph['root'] = 'a'


























# import numpy as np
#
# class Graph:
#     def __init__(self):
#         self.graph = {}
#
#     def populate(self, depth=2, actions=2, weight=10):
#         self.depth = depth
#         self.actions=actions
#         self.weight= weight
#         self.graph['root'] = ['A0',0]
#
#         for d in range(self.depth):
#             for a in range(self.actions):
#                 self.graph[chr(ord('A') + d) + str(a)] = \
#                     [chr(ord('A') + d + 1), self.weight]
#
# graph = Graph()
# graph.populate()
# graph.graph
#
# # graph = {}
# # graph['root'] = 'A'
#
# w = {}
# w['a0'] = [5, 10]
# w['a1'] = [5, 5]
# w['a2'] = [10, 10]
#
# g = {}
# g['root'] = 's0'
# g['s0'] = ['a0','a1','a2']
# g['a0'] = [['s1_00', w['a0'][0]],['s1_01', w['a0'][1]]]
# g['a1'] = [['s1_10', w['a1'][0]],['s1_11', w['a1'][1]]]
# g['a2'] = [['s1_20', w['a2'][0]],['s1_21', w['a2'][1]]]
