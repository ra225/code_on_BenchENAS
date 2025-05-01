## Installation

1. Download the source code.
```
$ git clone https://github.com/benchenas/BenchENAS.git
$ cd BenchENAS_linux_platform
```
2. Install sshpass.
```
$ sudo apt-get install sshpass
```
3. Install redis and start redis server.
```
$ sudo apt-get install redis-server
$ redis-server --protected-mode on
```
4. Installation of python third-party libraries.
```
$ pip install redis
$ pip install paramiko
```

## Running

1. Start the init_compute.py script to start the platform.

```
$ python main.py
```

## Running

2. Start the algorithm you would like to perform.

```
$ python init_compute.py
```

## Citation
```
@inproceedings{wang2023neural,
  title={A Neural Architecture Search Method using Auxiliary Evaluation Metric based on ResNet Architecture},
  author={Wang, Shang and Tang, Huanrong and Ouyang, Jianquan},
  booktitle={Proceedings of the Companion Conference on Genetic and Evolutionary Computation},
  pages={687--690},
  year={2023}
}
```
