import re
import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np

from WeightType import WeightType
from Graph import Graph
from Settings import Settings
from Service import Service
class Clustering:

    @staticmethod
    def compute_multiple_resolutions(graph, start, end, step, weight_type=WeightType.STRUCTURAL):

        clusters_results = []
        res_range = np.arange(start, end, step)

        for resolution in res_range:
            for _ in range(3):
                clusters, modularity = Clustering.community_detection_louvain(
                    graph, resolution=resolution)
                clusters_results.append((clusters, modularity, resolution))

        Clustering.write_modularity_and_services(clusters_results)

        return clusters_results


    @staticmethod
    def pre_process(graph, remove_weak_edges=False, remove_disconnected_sections=False):
        # TODO: could be optimized by caching already traversed nodes

        graph = graph.to_undirected()

        # Remove edges with weak weights. Could have a moderate impact on louvain due to the way it decides which community to choose
        edges_remove = []
        if remove_weak_edges:
            for edge in graph.edges:
                data = graph.get_edge_data(edge[0], edge[1])
                if data and data['weight_structural'] <= 0.3:
                    edges_remove.append((edge[0], edge[1]))

            for edge in edges_remove:
                graph.remove_edge(edge[0], edge[1])
                print(f"Removing edge (=0) {edge}")

        # Remove nodes that belong to a disconnected section consisting of less than [node_depth] nodes
        nodes_remove = []
        if remove_disconnected_sections:
            for node in graph.nodes():
                node_depth = 5
                edges = nx.dfs_edges(graph, source=node,
                                     depth_limit=node_depth)
                count = 0

                for edge in edges:
                    count += 1

                if count < node_depth:
                    nodes_remove.append(node)

            for node in nodes_remove:
                graph.remove_node(node)
                # print(f"Ignoring node (<{node_depth} dfs) {node}")

        return graph

    @staticmethod
    def community_detection_louvain(graph, resolution=1.0, weight_type=WeightType.STRUCTURAL):
        weight_type = weight_type.value
        graph = Graph.to_undirected(graph)

        # Lower resolution results in more clusters
        partition = community.best_partition(
            graph, weight=weight_type, resolution=resolution)
        values = [partition.get(node) for node in graph.nodes()]

        nodes = list(graph.nodes)
        clusters = {}
        for index, val in enumerate(values):
            if val not in clusters:
                clusters[val] = []

            clusters[val].append(nodes[index])

        print(f"Total Clusters: {len(clusters)}")

        # TODO : refactor and move this section to Graph.draw()
        # Relabel nodes from qualified name (package+classname) to classname for better graph visibility
        # This can cause problems if there are 2 classes with the same name on different packages
        # eg. com.blog.controllers.PostController and com.blog.admin.PostController
        h = graph.copy()
        mappings = {}

        cluster_distribution = [len(cluster) for cluster in clusters.values()]
        print(f"Cluster distribution by class count: {cluster_distribution}")
        modularity = community.modularity(partition, graph)
        print(f"Modularity: {modularity}")

        if Settings.DRAW:

            for index, node in enumerate(h.nodes()):
                curr_class_name = re.search(r'\.(\w*)$', str(node))
                if curr_class_name:
                    # TODO : repôr mais tarde
                    mappings[node] = f"{curr_class_name[1]}_{index}"
            h = nx.relabel_nodes(h, mappings)

            # Drawing of labels explained here - https://stackoverflow.com/questions/31575634/problems-printing-weight-in-a-networkx-graph
            sp = nx.spring_layout(h, weight=weight_type, seed=1)
            nx.draw_networkx(h, pos=sp, with_labels=True,
                             node_size=1000, font_size=12, node_color=values, cmap=plt.cm.tab10)

            # edge_weight_labels = dict(map(lambda x: (
            #     (x[0], x[1]),  round(x[2][str(weight_type)], 2) if x[2][weight_type] > 0 else ""), h.edges(data=True)))

            # nx.draw_networkx_edge_labels(
            #     h, sp, edge_labels=edge_weight_labels, font_size=7, alpha=1)

            plt.show()

        return clusters, modularity

    @staticmethod
    def write_modularity_and_services(clusters_results):
        with open('./clustering.txt', 'w+') as f:
            for clusters, modularity, resolution in clusters_results:
                f.write(f"Modularity {modularity}\n")
                f.write(f"Resolution {resolution}\n")

                for service_id, classes in clusters.items():
                    f.write(f"Service {service_id}\n")
                    for classe in classes:
                        f.write(f"{classe}\n")
                    f.write("\n")

                f.write(f"{100*'-'}\n")
                f.write(f"{100*'-'}\n")
                f.write(f"{100*'-'}\n\n")

                Clustering.write_services_to_file(
                    clusters, resolution)

    @staticmethod
    def write_services_to_file(clusters, resolution):
        # service_id, service
        services = Service.extract_services_from_clusters(clusters)

        Settings.create_id()

        with open(f"{Settings.DIRECTORY}/data/{Settings.PROJECT_NAME}/{Settings.PROJECT_NAME}_{Settings.ID}_K{Settings.K_TOPICS}_R{round(resolution,2)}", 'w+') as f:
            # print(15*"-")
            # print(15*"-")
            # print(f"Services for resolution: {resolution}")
            for service_id, service in services.items():
                service_id = f"\nService {service_id}\n"
                f.write(service_id)
                # print(service_id)
                for class_name in service.get_classes():
                    f.write(f"{class_name}\n")
                    # print(f"\t{class_name}")
                f.write("\n")
                # print("\n")
