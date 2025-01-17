#!/usr/bin/bash

# Evaluate experiment 01 - real runs
var=( 5 6 7 )
for number in "${var[@]}"
do
  echo "Evaluation of experiment 01 - real - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/real/runs/0$number" \
    --results_dir="/results/exp_01/real/runs/0$number"
done

# Evaluate experiment 01 - mixed runs
for number in "${var[@]}"
do
  echo "Evaluation of experiment 01 - mixed - custom - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/mixed/custom/runs/0$number" \
    --results_dir="/results/exp_01/mixed/custom/runs/0$number"

  echo "Evaluation of experiment 01 - mixed - controlnet - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/mixed/controlnet/runs/0$number" \
    --results_dir="/results/exp_01/mixed/controlnet/runs/0$number"
done

# Evaluate experiment 01 - synthetic runs
for number in "${var[@]}"
do
  echo "Evaluation of experiment 01 - synthetic - custom - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/synthetic/custom/runs/0$number" \
    --results_dir="/results/exp_01/synthetic/custom/runs/0$number"

  echo "Evaluation of experiment 01 - synthetic - controlnet - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/synthetic/controlnet/runs/0$number" \
    --results_dir="/results/exp_01/synthetic/controlnet/runs/0$number"
done

# Evaluate experiment 02 - real runs
var_ds=( 20 40 60 80 100 200 400 600 800 )
var_run=( 11 12 13 14 15 16 )
for ds in "${var_ds[@]}"
do
  for run_id in "${var_run[@]}"
  do
    echo "Evaluation of experiment 02 - real - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/real/ds_$ds/runs/$run_id" \
      --results_dir="/results/exp_02/real/ds_$ds/runs/$run_id"
  done
done

# Evaluate experiment 02 - mixed runs
var_run=( 14 15 16 )
for ds in "${var_ds[@]}"
do
  for run_id in "${var_run[@]}"
  do
    echo "Evaluation of experiment 02 - mixed - custom - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/mixed/custom/ds_$ds/runs/$run_id" \
      --results_dir="/results/exp_02/mixed/custom/ds_$ds/runs/$run_id"

    echo "Evaluation of experiment 02 - mixed - controlnet - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/mixed/controlnet/ds_$ds/runs/$run_id" \
      --results_dir="/results/exp_02/mixed/controlnet/ds_$ds/runs/$run_id"
  done
done