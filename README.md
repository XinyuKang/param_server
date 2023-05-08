# CS 262 Final Project - Parameter Server

## Usage

```bash
bash shell/script.sh
```

## File Stucture

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
├── result.ipynb
└── README.md
└── final_report.pdf
```

In this implementation, we use `main.py` to run the program. `param_server.py` contains shcheduler, server and worker implementation. `model.py` describes the deep learning model. `data_loader.py` is the data loader. `fault_tolarent.py` is the implementation of fault tolerance in this project. `test.py` is the unit test file. `shell/script.sh` is the shell script to run the program for experiments. Our training results used in our experiments are located in results/. `result.ipynb` is the notebook to generate the plots in our report.

## Set up environment
```pip install -r requirements.txt```

## Report 
Please see final_report.pdf for additional implementation details. 

