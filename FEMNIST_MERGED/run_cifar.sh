#!/bin/bash

echo $1
port=$2
cid=$(($1-1))
cid=$(($cid * 9))
echo $(($cid  + 1))
if [[ $1 -eq 0 ]]
then 
    echo $1
    hostname -I > ip_cifar_res.txt
    server="localhost"
    #while IFS=' ' read -r ip rest
    #do 
    #    echo "$ip"
    #    server="$ip"
    #done <ip_cifar_res.txt
    python3 server.py --server_address $server:$port --rounds 2 --min_num_clients 8 --min_sample_size 4 --model Net

else 
    sleep 20
    echo $1
    server="127.0.0.1"

    python3 client.py --server_address=$server:$port --cid $(($cid  + 0)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 1)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 2)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 3)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 4)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 5)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 6)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 7)) --model Net --device cpu &
    python3 client.py --server_address=$server:$port --cid $(($cid  + 8)) --model Net --device cpu 
fi
 
