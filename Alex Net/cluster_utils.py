#!/bin/bash
export TF_RUN_DIR="~/tf"

function terminate_cluster() {
    echo "Terminating the servers"
#    CMD="ps aux | grep -v 'grep' | grep 'python code_template' | awk -F' ' '{print $2}' | xargs kill -9"
    CMD="ps aux | grep -v 'grep' | grep -v 'bash' | grep -v 'ssh' | grep 'python startserver' | awk -F' ' '{print \$2}' | xargs kill -9"
    for i in `seq 0 3`; do
        ssh node-$i "$CMD"
    done
}