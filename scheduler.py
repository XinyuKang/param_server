import models
from consistent_hashing import ConsistentHash
from server import ParameterServer
import numpy as np
from worker import DataWorker


def Scheduler(num_servers, num_workers, hashes_per_server=50):
    model = models.LinearNet()
    key_values = model.get_weights()
    keys = np.array(list(key_values.keys()))
    values = [key_values[key] for key in keys]

    key_indices = {key: x for x, key in enumerate(keys)}

    # distributing weights across servers - do this using consistency hashing
    server_ids = ["server" + str(ind) for ind in range(num_servers)]
    hasher = ConsistentHash(keys, server_ids, hashes_per_server)
    servers = [
        ParameterServer.remote(
            keys[[key_indices[key] for key in hasher.get_keys_per_node()[serv]]],
            [values[key_indices[key]] for key in hasher.get_keys_per_node()[serv]],
        )
        for serv in server_ids
    ]

    # creating equal workers per server
    weight_assignments = hasher.get_keys_per_node()
    workers = [
        [
            DataWorker.remote(weight_assignments["server" + str(j)])
            for i in range(num_workers)
        ]
        for j in range(num_servers)
    ]

    return (
        hasher,
        servers,
        workers,
        keys,
        model,
        hasher.get_keys_per_node(),
        server_ids.copy(),
    )
