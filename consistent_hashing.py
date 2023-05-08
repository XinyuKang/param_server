import hashlib


class ConsistentHash:
    """Consistent Hashing class

    Attributes:
        keys_to_hash: a list of keys to hash
        nodes: a list of nodes
        hashes_per_node: number of hashes per node
        node_hashes: a dictionary of node to hashes
        node_tuples: a list of tuples of (node, hash)
        hash_assignments: a dictionary of hash to node
        node_assignments: a dictionary of node to hash
    """

    def __init__(self, keys_to_hash, nodes, hashes_per_node=1):
        """Initializes a consistent hashing object.

        Args:
            keys_to_hash: a list of keys to hash
            nodes: a list of nodes
            hashes_per_node: number of hashes per node
        """
        self.keys_to_hash = keys_to_hash
        self.nodes = nodes
        self.hashes_per_node = hashes_per_node
        self.node_hashes = {}
        self.node_tuples = []
        self.hash_assignments = {}
        self.node_assignments = {}
        for node in nodes:
            self.add_node(node)
        for key in keys_to_hash:
            self.add_key(key)

    def get_hash_key(self, key):
        """Gets the hash for a key.

        Args:
            key: a key

        Returns:
            a hash
        """
        # Gets the hash for a key
        return hashlib.sha256(str(key).encode()).hexdigest()

    def get_hash_node(self, node):
        """Gets multiple hashes for a node in a deterministic manner.

        Args:
            node: a node

        Returns:
            a list of hashes
        """
        # Gets multiple hashes for a node in a deterministic manner
        return [
            hashlib.sha256((str(node) + str(hash_iter)).encode()).hexdigest()
            for hash_iter in range(self.hashes_per_node)
        ]

    def assign_to_node(self, key):
        """Assigns a key to the correct node.

        Args:
            key: a key
        """
        key_hash = self.hash_assignments[key]
        self.node_assignments[key] = self.node_tuples[0][1]
        for h, node in self.node_tuples:
            if h > key_hash:
                self.node_assignments[key] = node
                break

    def reassign_keys(self):
        """Reassigns keys when a node has been added or deleted."""
        self.node_tuples = sorted(
            [x for x in self.node_tuples if x[1] in self.node_hashes],
            key=lambda x: x[0],
        )
        for key in self.node_assignments:  # Won't do anything if no node added yet
            self.assign_to_node(key)

    def add_key(self, key):
        """Adds a key.

        Args:
            key: Target key
        """
        self.hash_assignments[key] = self.get_hash_key(key)
        self.assign_to_node(key)

    def delete_key(self, key):
        """Deletes a key.

        Args:
            key: Target key
        """
        del self.hash_assignments[key]
        del self.node_assignments[key]

    def add_node(self, node):
        """Adds a node

        Args:
            node: Target node
        """
        hashes = self.get_hash_node(node)
        self.node_hashes[node] = hashes
        self.node_tuples.extend([(h, node) for h in hashes])
        self.reassign_keys()

    def delete_node(self, node):
        """Deletes a node

        Args:
            node: Target node
        """
        del self.node_hashes[node]
        self.nodes.remove(node)
        self.reassign_keys()

    def delete_node_and_reassign_to_others(self, node):
        """Deletes a node and reassigns its keys to other nodes

        Args:
            node: Target node
        """
        keys_to_reassign = self.get_keys_per_node()[node]
        del self.node_hashes[node]
        self.nodes.remove(node)
        self.node_tuples = [x for x in self.node_tuples if x[1] in self.node_hashes]
        for key in keys_to_reassign:
            self.delete_key(key)
            self.add_key(key)

    def get_key_to_node_map(self):
        """Returns dictionary of key -> node mappings

        Returns:
            a dictionary of key to node
        """
        return self.node_assignments

    def get_keys_per_node(self):
        """Returns dictionary of node -> keys mappings

        Returns:
            a dictionary of node to keys
        """
        map_dict = {node: [] for node in self.nodes}
        for key in self.node_assignments:
            map_dict[self.node_assignments[key]].append(key)
        return map_dict
