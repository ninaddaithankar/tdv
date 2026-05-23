#!/bin/bash

# Slurm Submission Script
# this is used to execute slurm scripts while also saving their contents. this helps prevent creating tons of different slurm scripts for different hpc compute servers and is most helpful for re-executing slurm scripts if jobs failed or other issues occured. It also logs the contents so you can track what you have submitted or rerun failed jobs easily

# Usage: ./submit_job.sh <job_config_script>

set -e  # Exit immediately if a command exits with a non-zero status

usage() {
    echo "Usage:"
    echo "  $0 <job_config_script>"
    echo "  $0 <HEADER_TYPE> <job_config_script>"
    echo "Examples:"
    echo "  $0 ./job_scripts/tdv.slurm"
    echo "  $0 my_cluster ./job_scripts/tdv.slurm"
    echo ""
    echo "HEADER_TYPE must match a file in job_scripts/slurm_headers/<HEADER_TYPE>.slurm"
    echo "Use 'none' to run with bash instead of sbatch."
    exit 1
}

# Initialize variables
HEADER_TYPE="none"
JOB_CONFIG_PATH=""

# Parse command-line arguments
if [ "$#" -eq 1 ]; then
    JOB_CONFIG_PATH="$1"
elif [ "$#" -eq 2 ]; then
    HEADER_TYPE="$1"
    JOB_CONFIG_PATH="$2"
else
    echo "Error: Incorrect number of arguments."
    usage
fi

# Check if a Conda environment is active
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No active Conda environment detected."
    echo "Please activate a Conda environment before submitting the job."
    exit 1
fi

# Optional: Print the active Conda environment
echo "Active Conda environment: $(basename "$CONDA_PREFIX")"

# Define the directory paths
HEADER_DIR="./job_scripts/slurm_headers"

# Path to Slurm headers
HEADER_FILE="${HEADER_DIR}/${HEADER_TYPE}.slurm"

# Check if the corresponding header file exists (skip for 'none')
if [ "$HEADER_TYPE" != "none" ] && [ ! -f "$HEADER_FILE" ]; then
    echo "Error: Slurm header file '$HEADER_FILE' does not exist."
    echo "Add a header file at '$HEADER_FILE' for your cluster, or use 'none' to run with bash."
    exit 1
fi

# Create a temporary file to hold the combined script
TEMP_SCRIPT=$(mktemp ./job_scripts/temp_script_XXXXXX.slurm)

# Ensure the temporary script is removed on exit
trap 'rm -f "$TEMP_SCRIPT"' EXIT

# Add the selected Slurm header if not 'none'
if [ "$HEADER_TYPE" != "none" ]; then
    cat "$HEADER_FILE" > "$TEMP_SCRIPT"
    echo "" >> "$TEMP_SCRIPT"  # Add a newline for separation
fi

# Append the job configuration script
cat "$JOB_CONFIG_PATH" >> "$TEMP_SCRIPT"

# Ensure the temporary script is executable
chmod +x "$TEMP_SCRIPT"

# Determine the execution method based on HEADER_TYPE
if [ "$HEADER_TYPE" != "none" ]; then
    # Run the Slurm script using sbatch and capture the output
    execution_output=$(sbatch "$TEMP_SCRIPT")
    execution_method="sbatch"
else
    # Run the script using bash and capture the output
    execution_output=$(bash "$TEMP_SCRIPT")
    execution_method="bash"
fi

# Create the log directory if it doesn't exist
mkdir -p ./logs/job_scripts

# Log file path
LOG_FILE="./logs/job_scripts/executed_slurm_script_contents.log"

# Append the combined script contents to the log file with delimiters, timestamp, and script name
{
    echo "--------------------------------------------------"
    echo "Executed on: $(date)"
    echo "Job Config Script: $(basename "$JOB_CONFIG_PATH")"
    echo "Header Type: $HEADER_TYPE"
    echo "Execution Method: $execution_method"
    echo "--------------------------------------------------"
    cat "$TEMP_SCRIPT"
    echo ""
} >> "$LOG_FILE"

# Echo the sbatch output along with the confirmation message
echo "Script '$JOB_CONFIG_PATH' executed using $execution_method."
