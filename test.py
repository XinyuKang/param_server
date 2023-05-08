import ray
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import models
from models import MLP, evaluate
from param_server import ParameterServer, Scheduler, Worker
from consistent_hashing import ConsistentHash


# Create a small synthetic dataset for testing
X = torch.randn(100, 784)
y = torch.randint(0, 2, (100, 1)).float()


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MLP()
        dataset = TensorDataset(X, y)
        self.test_loader = DataLoader(dataset, batch_size=32)

    def test_forward(self):
        output = self.model(X)
        self.assertEqual(output.shape, (100, 1))

    def test_get_weights(self):
        weights = self.model.get_weights()
        self.assertEqual(len(weights), 785)

    def test_get_gradients(self):
        keys = list(self.model.state_dict().keys())
        output = self.model(X)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(output, y)
        loss.backward()

        gradients = self.model.get_gradients(keys)
        for g, p in zip(gradients, self.model.parameters()):
            self.assertTrue(torch.allclose(torch.from_numpy(g), p.grad))

    def test_set_gradients(self):
        keys = list(self.model.state_dict().keys())
        output = self.model(X)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(output, y)
        loss.backward()

        original_gradients = [p.grad.detach().numpy() for p in self.model.parameters()]
        self.model.set_gradients(original_gradients)

        for i, p in enumerate(self.model.parameters()):
            self.assertTrue(
                torch.allclose(p.grad, torch.from_numpy(original_gradients[i]))
            )

    def test_evaluate(self):
        accuracy = evaluate(self.model, self.test_loader)
        self.assertTrue(0.0 <= accuracy <= 100.0)


ray.init()


@ray.remote
class MockWorker:
    def get_gradients(self):
        return ["w1", "w2"], [np.array([1.0, 1.0]), np.array([2.0, 2.0])]


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.keys = ["layer1.0.weight", "layer1.0.bias"]
        self.worker = Worker.remote(self.keys)

    def test_initialization(self):
        self.assertIsNotNone(self.worker)


class TestParameterServer(unittest.TestCase):
    def test_initialization(self):
        keys = ["w1", "w2"]
        values = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        ps = ParameterServer.remote(keys, values)
        self.assertIsNotNone(ps)

    def test_add_weight(self):
        keys = ["w1", "w2"]
        values = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        ps = ParameterServer.remote(keys, values)

        ray.get(ps.add_weight.remote("w3", torch.tensor([1.0, 1.0])))

        weights = ray.get(ps.get_weights.remote(keys + ["w3"]))
        self.assertEqual(len(weights), 3)

    def test_get_weights(self):
        keys = ["w1", "w2"]
        values = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        ps = ParameterServer.remote(keys, values)

        weights = ray.get(ps.get_weights.remote(keys))
        for w1, w2 in zip(values, weights):
            self.assertTrue(torch.all(w1.eq(w2)))


class TestScheduler(unittest.TestCase):
    def test_scheduler_initialization(self):
        num_servers = 3
        num_workers = 2
        hashes_per_server = 50

        hasher, servers, workers, keys, model, keys_per_node, server_ids = Scheduler(
            num_servers, num_workers, hashes_per_server
        )

        # Check if the correct number of servers and workers are created
        self.assertEqual(len(servers), num_servers)
        self.assertEqual(len(workers), num_servers)
        for worker_group in workers:
            self.assertEqual(len(worker_group), num_workers)

        # Check if the keys are distributed among the servers
        total_keys = 0
        for server_id in server_ids:
            total_keys += len(keys_per_node[server_id])
        self.assertEqual(total_keys, len(keys))

        # Check if the model is of the correct type
        self.assertIsInstance(model, models.MLP)

        # Check if the hasher is initialized correctly
        self.assertIsNotNone(hasher)
        self.assertEqual(len(hasher.node_assignments), len(keys))


class TestConsistentHash(unittest.TestCase):
    def setUp(self):
        self.nodes = ["node1", "node2", "node3"]
        self.keys = ["key1", "key2", "key3", "key4", "key5"]
        self.ch = ConsistentHash(self.keys, self.nodes)

    def test_initialization(self):
        self.assertIsNotNone(self.ch)
        self.assertEqual(len(self.ch.node_assignments), len(self.keys))

    def test_add_key(self):
        new_key = "key6"
        self.ch.add_key(new_key)
        self.assertIn(new_key, self.ch.node_assignments)

    def test_delete_key(self):
        key_to_remove = "key1"
        self.ch.delete_key(key_to_remove)
        self.assertNotIn(key_to_remove, self.ch.node_assignments)

    def test_add_node(self):
        new_node = "node4"
        self.ch.add_node(new_node)
        self.assertIn(new_node, self.ch.node_hashes)

    def test_delete_node(self):
        node_to_remove = "node1"
        self.ch.delete_node(node_to_remove)
        self.assertNotIn(node_to_remove, self.ch.node_hashes)

    def test_delete_node(self):
        node_to_remove = "node1"
        self.ch.delete_node(node_to_remove)
        self.assertNotIn(node_to_remove, self.ch.node_hashes)
        for key, node in self.ch.node_assignments.items():
            self.assertNotEqual(node, node_to_remove)

    def test_get_key_to_node_map(self):
        key_to_node_map = self.ch.get_key_to_node_map()
        self.assertEqual(len(key_to_node_map), len(self.keys))

    def test_get_keys_per_node(self):
        keys_per_node = self.ch.get_keys_per_node()
        self.assertEqual(len(keys_per_node), len(self.nodes))


if __name__ == "__main__":
    unittest.main()
