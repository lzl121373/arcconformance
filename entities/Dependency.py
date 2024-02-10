

class Dependency:

    def __init__(self, src, dst, relations):
        self.src = src
        self.dst = dst
        self.relations = relations
        weight = 0
        for key in relations:
            if(key == 'Call'):
                weight += relations[key]
        self.weight = weight

    def get_src(self):
        return self.src

    def get_dst(self):
        return self.dst

    def get_relations(self):
        return self.relations

    def get_weight(self):
        return self.weight