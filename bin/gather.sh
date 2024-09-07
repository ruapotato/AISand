#!/bin/bash

# Number of elements
NUM_ELEMENTS=13

# Number of processes to use for single and pair simulations
NUM_PROCESSES=5

# Number of instances for all elements simulation
NUM_ALL_INSTANCES=5

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

# Function to run all elements simulation
run_all() {
    local instance_id=$1
    local images_per_instance=$((IMAGES_PER_ELEMENT / NUM_ALL_INSTANCES))
    python gather.py --segment all --images $images_per_instance --process_id $instance_id --output_dir "all_elements_instance_$instance_id" &
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

# Run all elements simulation with 10 instances
for ((i=0; i<NUM_ALL_INSTANCES; i++)); do
    run_all $i
done

# Wait for all elements simulations to complete
wait

echo "All simulations complete."
