#!/bin/bash
echo "Server or client (0-server 1-client)?"
read cid
echo "Enter port number: "
read p_num
echo "Enter number of clients: "
read c_num
if [[ $cid -eq 0 ]]; then
echo "Enter Sparsity+Quantization+FedAdam as a single input of consective characters (combination of s, q, and f like sqf or qf to set only those switches to true) or n for default FLuID: "
read input
echo "Enter number of rounds: "
read rounds
else
echo "Enable quantization (q/n)? : "
read input
fi
# Initialize variables
s=false
q=false
f=false

# Loop through each character in the input
for char in $(echo $input | grep -o .); do
    case $char in
        s) s=true ;;
        q) q=true ;;
        f) f=true ;;
        n) echo "Default FLuID" ;;
        *) echo "Invalid input: $char"; exit 1 ;;
    esac
done
if [ "$s" = true ]; then
   echo "You have enabled Sparsity, enter a sparsity threshold: "
   read sparsity_threshold
fi
# Print the status of variables
echo "Sparsity: $s"
echo "Quantization: $q"
echo "FedAdam: $f"


factor=0.8
port=$p_num
cid=$cid
if [[ $cid -eq 0 ]]
then 
    hostname -I > ip.txt
    server="localhost"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip.txt
    sample=$(echo "scale=0; ($factor * $c_num + 0.5) / 1" | bc)
    command="python3 server.py --server_address $server:$port --rounds $rounds --min_num_clients $c_num --min_sample_size $sample --model Net"
    if [ "$s" = true ]; then
       command+=" --sparsity True --sparsity_threshold $sparsity_threshold"
    fi
    if [ "$q" = true ]; then
       command+=" --quantization True"
    fi
    if [ "$f" = true ]; then
       command+=" --fedadam True"
    fi 
    $command
else 
    sleep 10
    echo $1
    server="127.0.0.1"
    while IFS=' ' read -r ip rest
    do 
        echo "$ip"
        server="$ip"
    done <ip.txt

    for ((cid=1; cid<=$c_num; cid++)); do
    command="python3 client.py --server_address="$server":"$port" --cid=$((cid + 0)) --model Net --device gpu"
    if [ "$q" = true ]; then
       command+=" --quantization True"
    fi
    $command &
    done
fi
 
