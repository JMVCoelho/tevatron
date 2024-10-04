# for i in {0..7}
# do
#     JOB0_ID=$(sbatch -d afterok:731808 experiments/debug/run_inf_qgen_minicpm.sh $i | awk '{print $NF}')
#     echo "Submitted batch job $JOB0_ID for iteration $i"
#     JOB1_ID=$(sbatch -d afterok:$JOB0_ID experiments/debug/test_rr_run.sh $i | awk '{print $NF}')
#     echo "Submitted batch job $JOB1_ID for iteration $i"
# done



