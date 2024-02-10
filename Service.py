import itertools

class Service:

    new_id = itertools.count()

    def __init__(self, service_id=None, classes={}):
        if service_id == None:
            self.id = next(Service.new_id)
        else:
            self.id = service_id
        self.classes = set(classes)
        self.external_classes_dependencies = {}
        self.service_dependencies = {}

    def get_classes(self):
        return self.classes

    def set_classes(self, classes):
        self.classes = classes

    @staticmethod
    def extract_services_from_clusters(clusters):
        services = {}

        for cluster_id in clusters:
            service = Service(cluster_id, clusters[cluster_id])
            services[cluster_id] = service

        return services