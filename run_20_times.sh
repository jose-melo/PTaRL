#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_type> <dataname>"
    exit 1
fi

# Assign arguments to variables
MODEL_TYPE=$1
DATANAME=$2

# Run the command 20 times
for i in {1..20}
do
    echo "Execution $i: Running train_final_version.py with model_type=$MODEL_TYPE and dataname=$DATANAME"
    python train_final_version.py --model_type "$MODEL_TYPE" --dataname "$DATANAME"
done

