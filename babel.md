
## Script to run on login node to check cluster gpu status:

python /opt/cluster_tools/babel_contrib/tir_tool/gpu.py

## Helpful links

Babel wiki  
https://hpc.lti.cs.cmu.edu/wiki/index.php?title=BABEL

Vejo for monitoring gpus (needs to be in CMU network)  
http://babel-cluster.lti.cs.cmu.edu/vejo/

Other tips  
https://docs.google.com/presentation/d/1AgyKU72PrZ5O2JVbNB-7t6kI1C35UrQjVLZPSLuPv7k/edit#slide=id.p


## Useful comands

### Set global huggingface cache

e.g.,  
export TRANSFORMERS_CACHE=/data/datasets/hf_cache  

### Interactive job:  
srun --mem 8GB --pty bash  

can also specify other flags for GPU etc, see below.  

### Batch job  
sbatch script.sh  

script.sh header should follow like:  

---  

#!/bin/bash  
#SBATCH --job-name=<job_name>  
#SBATCH --output=logs/%x-%j.out  
#SBATCH -e logs/%x-%j.err  
#SBATCH --partition=general  
#SBATCH --cpus-per-task=12  
#SBATCH --mem=50G  
#SBATCH --gres=gpu:6000Ada:1  
#SBATCH --time=2-00:00:00  
#SBATCH --exclude=babel-4-28  

eval "$(conda shell.bash hook)"  
conda activate <env_name>  

---  

### Job dependencies:   

sbatch -d afterok:<job_id> new_script.sh  

new_script will be executed after job_id finishes successfully.   

https://hpc.nih.gov/docs/job_dependencies.html  


### Cancel jobs
scancel <slurm_id>  
scancel {<slurm_id_low>..<slurm_id_high>}  

### View job queue  

squeue  

squeue -u <user_id>   

squeue -o "%8i %1t %9u %9P %13b %3C %5m %10n %11l %.10L %.20S %.20e %R"  

squeue -o "%8i %1t %9u %9P %13b %3C %5m %10n %11l %.10L %.20S %.20e %R" -u <user_id>  

## Data

/home/<id> -> 100GB  
/data/user_data/<id> -> 500GB. Access only through compute nodes, not login  
/scratch -> compute only, for temporary files  

/data/models  
/data/datasets  
have global models and datasets for the whole cluster (autofs).  


## May be helpful to check docs  

jupyter notebooks with gpu / compute nodes  
job arrays slurm   