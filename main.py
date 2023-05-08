import ray
import argparse
import param_server
import numpy as np
from time import time
import models
import data_loader
import json

parser = argparse.ArgumentParser(
    description="synchronous distributed linear regression"
)
parser.add_argument(
    "-ns", "--num_servers", type=int, help="an integer for the number of servers to use"
)
parser.add_argument(
    "-nw",
    "--num_workers",
    type=int,
    help="an integer for the number of workers to use per server",
)
parser.add_argument(
    "-o", "--output_path", type=str, help="output path path for results"
)
parser.add_argument(
    "-i",
    "--num_iterations",
    type=int,
    default=500,
    help="an integer  for training iterations",
)
parser.add_argument(
    "-c",
    "--checkpoint",
    type=int,
    default=1,
    help="an integer for checkpointing iteration",
)
parser.add_argument(
    "-f",
    "--do_failure_test",
    type=int,
    default=0,
    help="to do failure test set this to 1",
)
parser.add_argument(
    "-fs",
    "--server_to_fail",
    type=str,
    default="server4",
    help="server id to kill for failure test",
)
parser.add_argument(
    "-f_iter",
    "--iteration_to_fail",
    type=int,
    default=60,
    help="iteration for the server to kill",
)
parser.add_argument(
    "-ev_int",
    "--eval_interval",
    type=int,
    default=5,
    help="Interval at which we evaluate",
)
args = parser.parse_args()

iterations = args.num_iterations  # total iterations
num_workers = args.num_workers  # number of workers per server
num_servers = args.num_servers  # number of servers
checkpoint = (
    args.checkpoint
)  # sync workers and checkpoint weights after these many iterations

eval_interval = args.eval_interval  # evaluate and get accuracy every 5 iterations
hashes_per_server = 100
lr = 0.1

# parameters for filure analysis
do_failure_test = args.do_failure_test
failure_iter = args.iteration_to_fail
failure_server = args.server_to_fail

# timing checking
time_per_iteration = []
time_per_gradient_push = []


# accuracy
accuracy_per_iteration = []

if __name__ == "__main__":
    (
        hasher,
        servers,
        workers,
        keys,
        model,
        weight_assignments,
        server_ids,
    ) = param_server.Scheduler(num_servers, num_workers, hashes_per_server)
    ray.init(ignore_reinit_error=True)

    print(
        "average weight load per server: ",
        np.mean([len(weight_assignments[key]) for key in weight_assignments]),
    )
    test_loader = data_loader.get_data_loader()[1]

    print("Running synchronous parameter server training.")

    # we need to get a new keys order because we are not assuming a ordering in keys
    current_weights = []
    keys_order = []

    for j in range(num_servers):
        keys_order.extend(weight_assignments[server_ids[j]])
        current_weights.append(
            ray.get(servers[j].get_weights.remote(weight_assignments[server_ids[j]]))
        )
    curr_weights_ckpt = current_weights.copy()

    for i in range(iterations):
        start_i = time()

        # fail a server at a particular iteration
        if do_failure_test == 1 and i == failure_iter:
            server_ids_old = server_ids.copy()
            weight_assignments_old = weight_assignments.copy()
            # Define parameters that will need to be moved
            failure_params = weight_assignments[failure_server]
            # Delete server from hash ring and reassign params
            hasher.delete_node(failure_server)
            weight_assignments = hasher.get_keys_per_node()
            # Update servers and workers
            num_servers -= 1
            server_ind = server_ids.index(failure_server)
            server_ids = server_ids[0:server_ind] + server_ids[server_ind + 1 :]
            servers = servers[0:server_ind] + servers[server_ind + 1 :]
            workers = workers[0:server_ind] + workers[server_ind + 1 :]
            # Add each relevant parameter to its new server
            server_dict = {server_ids[x]: servers[x] for x in range(len(server_ids))}
            for ind, param in enumerate(failure_params):
                server_dict[hasher.get_key_to_node_map()[param]].add_weight.remote(
                    param, curr_weights_ckpt[server_ind][ind]
                )
            for server in server_ids:
                sid = server_ids_old.index(server)
                for ind, param in enumerate(weight_assignments_old[server]):
                    server_dict[server].add_weight.remote(
                        param, curr_weights_ckpt[sid][ind]
                    )
            # Update these parameters for each worker to make them trainable
            [
                workers[j][idx].update_trainable.remote(
                    weight_assignments[server_ids[j]]
                )
                for idx in range(num_workers)
                for j in range(num_servers)
            ]
            current_weights = curr_weights_ckpt.copy()
            [
                workers[j][idx].update_weights.remote(keys_order, *current_weights)
                for idx in range(num_workers)
                for j in range(num_servers)
            ]

        # Evaluate model every eval_interval iterations.
        if i % eval_interval == 0:
            model.set_weights(keys_order, current_weights)
            accuracy = models.evaluate(model, test_loader)
            accuracy_per_iteration.append(accuracy)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

        # sync all weights on workers
        if i % args.checkpoint == 0:
            curr_weights_ckpt = current_weights.copy()

            # update weights on all workers
            [
                workers[j][idx].update_weights.remote(keys_order, *current_weights)
                for idx in range(num_workers)
                for j in range(num_servers)
            ]

        start_g = time()

        # use local cache of weights and get gradients from workers
        gradients = [
            [workers[j][idx].compute_gradients.remote() for idx in range(num_workers)]
            for j in range(num_servers)
        ]

        # Need to update key order if we've had a failure (can't do it before)
        if i == failure_iter:
            keys_order = []
            for j in range(num_servers):
                keys_order.extend(weight_assignments[server_ids[j]])

        # Updates gradients to specfic parameter servers
        current_weights_t = [
            servers[j].apply_gradients.remote(
                weight_assignments[server_ids[j]], lr, *gradients[j]
            )
            for j in range(num_servers)
        ]
        current_weights = ray.get(current_weights_t)

        end = time()
        time_per_iteration.append(end - start_i)
        time_per_gradient_push.append(end - start_g)

    # Clean up Ray resources and processes before the next example.
    ray.shutdown()

    print("Final accuracy is {:.1f}.".format(np.max(accuracy_per_iteration)))
    print("mean total time taken: ", np.mean(time_per_iteration[1:]))
    print("std total time taken: ", np.std(time_per_iteration[1:]))
    print(
        "mean total time taken per gradient update step: ",
        np.mean(time_per_gradient_push[1:]),
    )
    print(
        "std total time taken per gradient update step: ",
        np.std(time_per_gradient_push[1:]),
    )

    dict1 = {
        "total_time": time_per_iteration,
        "gradient_time": time_per_gradient_push,
        "accuracy": accuracy_per_iteration,
    }

    json.dump(dict1, open(args.output_path, "w"))
