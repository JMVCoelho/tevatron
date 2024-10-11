gen=gen17
model=/data/user_data/jmcoelho/models/query_generators/minicpm-2b-stf-bf6-marco-query-generator
episode=1

for i in {0,1,2,3,4,5,7}
do  
    # JOB0_ID=$(sbatch -d afterok:733646 experiments/debug/run_inf_qgen_minicpm.sh "$i" "$model" "$gen" "$episode" | awk '{print $NF}')
    # echo "Submitted batch job $JOB0_ID for iteration $i"
    JOB1_ID=$(sbatch experiments/debug/test_rr_run.sh "$i" "$gen" | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID for iteration $i"
done

#gen14: dpo-e2, 2.4m queries
#gen15: dpo-e1, 2.4m queries

# gen=gen15
# model=/data/user_data/jmcoelho/models/query_generators/minicpm-2b-stf-bf6-gpt4-query-generator-dpo-e1
# episode=1
# for i in {0..7}
# do  
#     JOB0_ID=$(sbatch experiments/debug/run_inf_qgen_minicpm.sh "$i" "$model" "$gen" "$episode" | awk '{print $NF}')
#     echo "Submitted batch job $JOB0_ID for iteration $i"    
# done

