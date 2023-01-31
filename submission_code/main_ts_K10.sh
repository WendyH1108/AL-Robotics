for seed in 0
do
    for task in brightness canny_edges dotted_line fog glass_blur identity impulse_noise motion_blur rotate scale shear shot_noise spatter stripe translate zigzag
    do
        for i in 0 1 2 3 4 5 6 7 8 9
        do
        echo ${i}_${task}_K10
        python main_ts.py --target_label $i --seed $seed --target_task $task --num_iter 1 --L 2.0 --epoch_num 2 --N_lowerBound 40 --debug_num 2000 --vAvg 20 --N_target 500 --rerun 0 --sparsity 10 --K 10 # non-linear case
        done
    done
done