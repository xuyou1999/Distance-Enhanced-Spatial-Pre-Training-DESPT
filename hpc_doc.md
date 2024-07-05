# Compute Facilities for TU/e Members

## Overview
TU/e members have access to multiple compute facilities. Detailed information can be found on the [SharePoint page](https://tuenl.sharepoint.com/sites/intranet-LIS/SitePages/Compute-Facilities.aspx).

## Compute Facilities

### TU/e HPC Cluster

#### Account Setup
To use the TU/e HPC Cluster, obtain an account with approval from your supervisor. Once your account is created, you can access the cluster using the OnDemand interface, which allows you to:
- Submit jobs
- Monitor job status
- Access the terminal of compute nodes

#### Documentation and Instructions
Detailed instructions on using the HPC cluster are available [here](https://supercomputing.tue.nl/documentation/steps/access/openondemand/).

#### Using the HPC Cluster
For intensive data projects, you may use either an interactive Jupyter Hub session or submit batch jobs via the SLURM job scheduler with a script. Below is an example script:

```bash
#!/bin/bash

#SBATCH --job-name=experiment
#SBATCH --output=slurm_log/experiment_output_%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=first.last@student.tue.nl

# Load modules or software if needed
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Execute the script or command
python execution_tune.py
```

#### Tips for Writing the Script
- Ensure necessary modules and software are loaded using `module avail`.
- Use the `--mail-user` option to receive email notifications.
- Save job output with the `--output` option for debugging.
- The MCS partition often provides better performance.
- Jupyter Hub has a maximum session time limit of 4 hours.

#### Node Specifications
HPC nodes have different specifications. Choose based on your requirements, such as nodes with GPUs for GPU tasks. Specifications can be found [here](https://supercomputing.tue.nl/documentation/specifications/).

#### Job Submission
Submit your job using the terminal app in the OnDemand interface with the following command:
```bash
sbatch script.sh
```
Monitor job status under the "Jobs" - "Active Jobs" tab.

#### Important Notes
- Requires VPN connection to TU/e network or on-campus WiFi.
- No internet access; save results on the cluster storage and download to your local machine.
- Plan jobs considering possible queue times.
- Use resources wisely as the cluster is shared.
- For multiple GPUs, ensure your code can utilize multiple GPUs (MIGs).

### Dutch National Supercomputer Snellius

#### Account Setup
To use Snellius, obtain an account from SURFsara as detailed [here](https://tuenl.sharepoint.com/sites/intranet-LIS/SitePages/Step-5--Dutch-National-Supercomputer.aspx).

#### Using Snellius
Submit jobs using the SLURM job scheduler or Jupyter Hub. TU/e members have reserved nodes on Snellius, available for free. However, SLURM jobs require credits, which can be requested through an online meeting with LIS.

#### Documentation
Further details can be found [here](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius).

#### Important Notes
- Accessible from anywhere, no VPN required.
- Internet access available; upload results directly to cloud storage.
- Jupyter Hub has a maximum session time limit of 2 hours.

For questions or further assistance, please contact the LIS support team.