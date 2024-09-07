#!/bin/bash

# Number of elements
NUM_ELEMENTS=13

# Number of processes to use
NUM_PROCESSES=4

# Number of images to generate per element/combination
IMAGES_PER_ELEMENT=1000

# Function to run a single element simulation
run_single() {
    local start=$1
    local end=$2
    local process_id=$3
    python gather.py --segment single --start $start --end $end --images $IMAGES_PER_ELEMENT --process_id $process_id &
}

# Function to run pair simulations
run_pair() {
    local start=$1
    local end=$2
    local process_id=$3
    python gather.py --segment pair --start $start --end $end --images $IMAGES_PER_ELEMENT --process_id $process_id &
}

# Run single element simulations
elements_per_process=$((NUM_ELEMENTS / NUM_PROCESSES))
for ((i=0; i<NUM_PROCESSES; i++)); do
    start=$((i * elements_per_process))
    end=$((start + elements_per_process))
    if [ $i -eq $((NUM_PROCESSES - 1)) ]; then
        end=$NUM_ELEMENTS
    fi
    run_single $start $end $i
done

# Wait for single element simulations to complete
wait

# Run pair simulations
for ((i=0; i<NUM_PROCESSES; i++)); do
    start=$((i * elements_per_process))
    end=$((start + elements_per_process))
    if [ $i -eq $((NUM_PROCESSES - 1)) ]; then
        end=$NUM_ELEMENTS
    fi
    run_pair $start $end $i
done

# Wait for pair simulations to complete
wait

# Run all elements simulation
python gather.py --segment all --images $IMAGES_PER_ELEMENT --process_id 0

echo "All simulations complete."
