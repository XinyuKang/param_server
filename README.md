# Parameter Server distributed training

## Implementation

```bash
.
├── mnist_data
│   └── ...
├── model_ops
│   └── resnet.py
├── main.py
├── nn_train.py
├── server.py
├── worker.py
└── README.md
```

## Usage

Here is a simple example of how to use this code with a server and two workers.

### Start Parameter Server

```bash
python server.py
```

### Start Worker

```bash
python worker.py --rank 1
python worker.py --rank 2
```