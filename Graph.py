import sys
import networkx as nx
import pygraphviz as pygv

from Settings import Settings
from WeightType import WeightType
import logging

class Graph:

    @staticmethod
    def create_dependencies(classes, dependencies, graph):
        for dependency in dependencies:
            src_id = dependency.get_src()
            dst_id = dependency.get_dst()

            if(src_id == dst_id):
                continue

            graph.add_edge(src_id, dst_id, weight_structural=dependency.get_weight())

        return graph

    @staticmethod
    def DrawInitialGraph(G):
        global DrawG
        scale = 50
        DrawG = pygv.AGraph(directed=True, strict='true', splines='true')

        for i in G.nodes():
            DrawG.add_node(i, shape='circle', label=G.nodes[i]['name'])

        for i in G.edges():
            DrawG.add_edge(i[0], i[1], color='black', label=G[i[0]][i[1]]['weight_structural'], fontsize='8')

        DrawG.layout(prog='dot')
        DrawG.draw(Settings.DATA_DIRECTORY + '\\' + Settings.PROJECT_NAME + '\\' + Settings.PROJECT_NAME + '_graph.png')


    @staticmethod
    def normalize_values(graph, dependency_type):
        min = sys.maxsize
        max = -sys.maxsize - 1
        # max_src, max_dst = 0, 0
        for src, dst in graph.edges():
            edge_data = graph.get_edge_data(src, dst)
            if dependency_type in edge_data:
                edge = edge_data[dependency_type]
                if edge < min:
                    min = edge
                if edge > max:
                    max = edge
                    # max_src = src
                    # max_dst = dst

        for src, dst in graph.edges():
            edge_data = graph.get_edge_data(src, dst)

            if dependency_type in edge_data:
                edge = edge_data[dependency_type]
                edge = (edge - min) / (max - min)

                graph[src][dst][dependency_type] = edge

        return graph

    @staticmethod
    def to_undirected(graph):
        """ Take into consideration the weighted edges and select the max.
        nx.to_undirected() keeps the last weight """
        G = nx.Graph()
        new_edges = {}
        for src, dst in graph.copy().edges():
            if(src, dst) in new_edges or (dst, src) in new_edges:
                continue

            edge_data_1 = graph.get_edge_data(src, dst)
            edge_data_2 = graph.get_edge_data(dst, src)
            G.add_edge(src, dst, weight=0)
            if(edge_data_1):
                G[src][dst]['weight'] += edge_data_1['weight_structural']
            if (edge_data_2):
                G[src][dst]['weight'] += edge_data_2['weight_structural']

        return G


