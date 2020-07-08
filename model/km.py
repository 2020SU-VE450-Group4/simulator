#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 00:39:41 2020

@author: liujiachen
"""


#import os
import pandas as pd
#import json

class KMNode(object):
    def __init__(self, id, exception=0, match=None, visit=False):
        self.id, self.exception = id, exception
        self.match, self.visit = match, visit

class KuhnMunkres(object):
    def __init__(self):
        self.matrix = None
        self.x_nodes, self.y_nodes = [], []
        self.minz = float('inf')
        self.x_length, self.y_length = 0, 0
        self.index_x, self.index_y = 0, 1
        self.zero_threshold = 1e-3
        
    def set_matrix(self, x_y_values):
        xs = set([x for x, y, value in x_y_values])
        ys = set([y for x, y, value in x_y_values])
        if len(xs) > len(ys):
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs
        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length, self.y_length = len(xs), len(ys)
        self.matrix = [[0]*self.y_length]*self.x_length
        for row in x_y_values:
            x, y, value = row[self.index_x], row[self.index_y], row[2]
            x_index, y_index = x_dic[x], y_dic[y]
            self.matrix[x_index][y_index] = value
        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i][:])

    def km(self):
        for i in range(self.x_length):
            while True:
                self.minz = float('inf')
                self.set_false(self.x_nodes)
                self.set_false(self.y_nodes)
                if self.dfs(i):
                    break
                self.change_exception(self.x_nodes, -self.minz)
                self.change_exception(self.y_nodes, self.minz)

    def dfs(self, cur_x_node):
        match_list = []
        while True:
            x_node = self.x_nodes[cur_x_node]
            x_node.visit = True
            for j in range(self.y_length):
                y_node = self.y_nodes[j]
                if not y_node.visit:
                    t = x_node.exception + y_node.exception - self.matrix[cur_x_node][j]
                    if abs(t) < self.zero_threshold:
                        y_node.visit = True
                        match_list.append((cur_x_node, j))
                        if y_node.match is None:
                            self.set_match_list(match_list)
                            return True
                        else:
                            cur_x_node = y_node.match
                            break
                    else:
                        if t >= self.zero_threshold:
                            self.minz = min(self.minz, t)
            else:
                return False
            
    def set_match_list(self, match_list):
        for i, j in match_list:
            x_node, y_node = self.x_nodes[i], self.y_nodes[j]
            x_node.match, y_node.match = j, i
            
    def set_false(self, nodes):
        for node in nodes:
            node.visit = False
            
    def change_exception(self, nodes, change):
        for node in nodes:
            if node.visit:
                node.exception += change
    def get_connect_result(self):
        result = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            y_node = self.y_nodes[j]
            x_id, y_id = x_node.id, y_node.id
            value = self.matrix[i][j]
            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            result.append((x_id, y_id, value))
        return result