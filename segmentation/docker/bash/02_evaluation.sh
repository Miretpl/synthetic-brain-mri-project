#!/bin/bash

# Evaluate experiment 01 - real runs
var=( 1 2 3 )
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

  echo "Evaluation of experiment 01 - mixed - spade - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/mixed/spade/runs/0$number" \
    --results_dir="/results/exp_01/mixed/spade/runs/0$number"

  echo "Evaluation of experiment 01 - mixed - pix2pix - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/mixed/pix2pix/runs/0$number" \
    --results_dir="/results/exp_01/mixed/pix2pix/runs/0$number"
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

  echo "Evaluation of experiment 01 - synthetic - spade - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/synthetic/spade/runs/0$number" \
    --results_dir="/results/exp_01/synthetic/spade/runs/0$number"

  echo "Evaluation of experiment 01 - synthetic - pix2pix - run_id 0$number"
  python ./code/evaluate.py \
    --runs_dir="/models/exp_01/synthetic/pix2pix/runs/0$number" \
    --results_dir="/results/exp_01/synthetic/pix2pix/runs/0$number"
done

# Evaluate experiment 02 - mixed big runs
#var_ds=( 4000 6000 8000 10000 12000 )
#for ds in "${var_ds[@]}"
#do
#  for run_id in "${var[@]}"
#  do
#    echo "Evaluation of experiment 02 - mixed big - custom - ds $ds - run_id $run_id"
#    python ./code/evaluate.py \
#      --runs_dir="/models/exp_02/mixed/extra_exp/custom/ds_$ds/runs/$run_id" \
#      --results_dir="/results/exp_02/mixed/extra_exp/custom/ds_$ds/runs/$run_id"
#
#    echo "Evaluation of experiment 02 - mixed big - controlnet - ds $ds - run_id $run_id"
#    python ./code/evaluate.py \
#      --runs_dir="/models/exp_02/mixed/extra_exp/controlnet/ds_$ds/runs/$run_id" \
#      --results_dir="/results/exp_02/mixed/extra_exp/controlnet/ds_$ds/runs/$run_id"
#
#    echo "Evaluation of experiment 02 - mixed big - spade - ds $ds - run_id $run_id"
#    python ./code/evaluate.py \
#      --runs_dir="/models/exp_02/mixed/extra_exp/spade/ds_$ds/runs/$run_id" \
#      --results_dir="/results/exp_02/mixed/extra_exp/spade/ds_$ds/runs/$run_id"
#
#    echo "Evaluation of experiment 02 - mixed big - pix2pix - ds $ds - run_id $run_id"
#    python ./code/evaluate.py \
#      --runs_dir="/models/exp_02/mixed/extra_exp/pix2pix/ds_$ds/runs/$run_id" \
#      --results_dir="/results/exp_02/mixed/extra_exp/pix2pix/ds_$ds/runs/$run_id"
#  done
#done

# Evaluate experiment 02 - real runs
var_ds=( 20 40 60 80 100 200 400 600 800 )
var_run=( 1 2 3 4 5 6 )
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
for ds in "${var_ds[@]}"
do
  for run_id in "${var[@]}"
  do
    echo "Evaluation of experiment 02 - mixed - custom - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/mixed/custom/ds_$ds/runs/$run_id" \
      --results_dir="/results/exp_02/mixed/custom/ds_$ds/runs/$run_id"

    echo "Evaluation of experiment 02 - mixed - controlnet - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/mixed/controlnet/ds_$ds/runs/$run_id" \
      --results_dir="/results/exp_02/mixed/controlnet/ds_$ds/runs/$run_id"

    echo "Evaluation of experiment 02 - mixed - spade - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/mixed/spade/ds_$ds/runs/0$run_id" \
      --results_dir="/results/exp_02/mixed/spade/ds_$ds/runs/0$run_id"

    echo "Evaluation of experiment 02 - mixed - pix2pix - ds $ds - run_id $run_id"
    python ./code/evaluate.py \
      --runs_dir="/models/exp_02/mixed/pix2pix/ds_$ds/runs/0$run_id" \
      --results_dir="/results/exp_02/mixed/pix2pix/ds_$ds/runs/0$run_id"
  done
done