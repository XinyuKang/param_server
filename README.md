# parameter server

## Usage

```bash
bash shell/script.sh
```

## Implementation

```bash
.
├── results
│   └── ...
├── shell
│   └── ...
├── main.py
├── fault_tolarent.py
├── data_loader.py
├── model.py
├── param_server.py
├── test.py
└── README.md
```

In this implementation, we use `main.py` to run the program. `param_server.py` contains shcheduler, server and worker implementation. `model.py` describes the deep learning model. `data_loader.py` is the data loader. `fault_tolarent.py` is the implementation of fault tolerance in this project. `test.py` is the unit test file. `shell/script.sh` is the shell script to run the program for experiments.
