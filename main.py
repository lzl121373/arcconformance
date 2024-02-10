
import argparse
from Settings import Settings
import os
import Utils as utils
import json
import networkx as nx

from entities.Dependency import Dependency
from Graph import Graph
from WeightType import WeightType
from Clustering import Clustering
import LDA as lda
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def main():
    project_name = 'jabref'
    project_path = 'V:\\papercode\\jabref\\src\\main\\java'
    Settings.PROJECT_PATH = project_path
    Settings.PROJECT_NAME = project_name

    # create_logging_folders(project_name)

    clusters_results = identify_clusters_in_project()


def identify_clusters_in_project():
    temp_json_location = f'{Settings.DATA_DIRECTORY}\\{Settings.PROJECT_NAME}\\{Settings.PROJECT_NAME}-class.json'

    # utils.execute_parser(project_path)

    # Read parsed document
    parsed_raw_json = {}
    with open(temp_json_location) as json_file:
            parsed_raw_json = json.load(json_file)

    classes, dependencies, class_docs = extract_graph_information_from_json(parsed_raw_json)

    graph = nx.DiGraph()
    graph = Graph.create_dependencies(classes, dependencies, graph)
    # Graph.DrawInitialGraph(graph)
    graph = Graph.normalize_values(graph, WeightType.STRUCTURAL.value)

    # graph = Clustering.pre_process(
    #     graph, remove_weak_edges=False, remove_disconnected_sections=True)

    clusters_results = []

    if Settings.RESOLUTION:
        clusters, modularity = Clustering.community_detection_louvain(
            graph, resolution=Settings.RESOLUTION)
        clusters_results.append((clusters, modularity, Settings.RESOLUTION))
        Clustering.write_modularity_and_services(clusters_results)
    else:
        clusters_results = Clustering.compute_multiple_resolutions(
            graph, start=0.5, end=1.1, step=0.1)

    max_modularity_cluster = max(clusters_results, key=lambda x: x[1])
    lda.apply_lda_to_clusters(clusters, class_docs)
    return clusters_results



def create_logging_folders(project_name):
    project_dir = f"{Settings.DIRECTORY}\\data\\{project_name}"
    directories = [project_dir]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def extract_graph_information_from_json(json_dict):
    classes = json_dict['variables']
    dependencies = []
    for cell in json_dict['cells']:
        src = cell['src']
        dst = cell['dest']
        values = cell['values']
        new_dependency = Dependency(src, dst, values)
        if(new_dependency.get_weight() != 0):
            dependencies.append(new_dependency)
    docs = []
    for clas in json_dict['classes']:
        doc = []
        doc.append(clas['rawname'])
        if len(clas['methods']) != 0:
            doc.extend(clas['methods'])
        if len(clas['variables']) != 0:
            doc.extend(clas['variables'])
        docs.append(doc)
    return classes, dependencies, docs





if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
