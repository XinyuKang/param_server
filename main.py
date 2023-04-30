import argparse
import os
import torch
import torch.distributed as dist
from worker import *
from server import *
from torchvision import datasets, transforms
        

def add_args(parser):
    parser.add_argument('--enable-gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='TBS',
                        help='testing batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train')
    parser.add_argument('--max-num-step', type=int, default=10000, metavar='MS',
                        help='the maximum number of iterations')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='EF',
                        help='how frequently the model should be evaluated')
    parser.add_argument('--train-dir', type=str, default='output/models/', metavar='TD',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--rank', type=int, default=0, metavar='RK',
                        help='the rank of this worker, default to server')
    parser.add_argument('--world-size', type=int, default=2, metavar='WS',
                        help='number of total workers')
    parser.add_argument('--server-addr', type=str, default='localhost', metavar='SA',
                        help='address of the server')
    parser.add_argument('--server-port', type=str, default='65000', metavar='SA',
                        help='port of the server')
    
    args = parser.parse_args()
    return args

  
    
if __name__ == "__main__":
    # rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])
    # master_addr = os.environ['SERVER_ADDR']
    # master_port = os.environ['SERVER_PORT']

    

    args = add_args(argparse.ArgumentParser(description=''))
    rank = args.rank
    os.environ['RANK'] = str(rank)
    world_size = args.world_size
    os.environ['WORLD_SIZE'] = str(world_size)
    master_addr = args.server_addr
    os.environ['MASTER_ADDR'] = master_addr
    master_port = args.server_port
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)

    # prepare data
    training_set = datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.test_batch_size, shuffle=True)

    device = torch.device("cuda" if args.enable_gpu else "cpu")

    kwargs_server = {
                'world_size':world_size,
                'batch_size':args.batch_size, 
                'lr':args.lr, 
                'max_epochs':args.epochs, 
                'momentum':args.momentum, 
                'eval_freq':args.eval_freq, 
                'train_dir':args.train_dir, 
                'max_num_step':args.max_num_step, 
                'device':device}

    kwargs_worker = {
                'rank':rank,
                'batch_size':args.batch_size, 
                'lr':args.lr, 
                'max_epochs':args.epochs, 
                'momentum':args.momentum, 
                'eval_freq':args.eval_freq, 
                'train_dir':args.train_dir, 
                'max_num_step':args.max_num_step, 
                'device':device}

    if rank == 0:
        param_server = Server(**kwargs_server)
        param_server.build_model(num_classes=10)
        print("This is the server, the world size is {}, cur step: {}".format(param_server.world_size, param_server.cur_step))
        param_server.start()
        print("Server is done sending messages to workers!")
    else:
        worker = Worker(**kwargs_worker)
        worker.build_model(num_classes=10)
        print("This is the worker: {} in all {} workers".format(worker.rank, worker.world_size-1))
        worker.train(train_loader=train_loader, test_loader=test_loader)