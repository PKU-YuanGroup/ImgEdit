

class NodeScheduler:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def 
    
    def schedule(self, task):
        for node in self.nodes:
            if node.can_handle(task):
                node.handle(task)
                return
