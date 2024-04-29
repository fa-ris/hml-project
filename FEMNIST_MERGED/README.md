## WELCOME to Beyond FLuID: Exploring Sparsity and Algorithm Variability in Federated Learning
What follows is a step-by-step guide on how to setup your client/s and the server. 

## Clone this repo


```bash
$ git clone 
```
## Getting your environment ready:

We recommend setting up a Python (Version 3.9) Virtual Environment using the requirements.txt file provided. Use the following command:
```bash
$ python3.9 -m venv myenv
$ source myenv/bin/activate
$ pip install -r requirements.txt
```

## Setting up the server

We have provided a simple bash script that can be used to run the server and client along with all the other options we provide. Some parameters can be directly changed in the bash script like the number of rounds. The rest are user inputs. 
The user is prompted to enter the port number, number of clients and whether to enable sparsity, quantization or change the optimization strategy to FedAdam. Users can, for example, run the following commands in different terminals to enable sparsity optimization:
```bash
$ bash ./run_femnist.sh
  Server or client (0-erver 1-client)?
  0
  Enter port number: 
  8005
  Enter number of clients: 
  5
  Enter Sparsity+Quantization+FedAdam as a single input of consective charactiers (combination of s, q, and f like sqf or qf to set only those switches to true): 
  s
  You have enabled Sparsity, enter a sparsity threshold: 
  0.5

$ bash ./run_femnist.sh
  Server or client (0-erver 1-client)?
  1
  Enter port number:
  8005
  Enter number of clients:
  5
  Enable quantization (q/n)?
  n
```
For the clients, the only option that can be enabled is quantization. The rest are server side options. 
