for seed in 0
do
    for task in brightness canny_edges dotted_line fog glass_blur identity impulse_noise motion_blur rotate scale shear shot_noise spatter stripe translate zigzag
    do
        for i in 0 1 2 3 4 5 6 7 8 9
        do
        echo ${i}_${task}
        python main_ts_linear.py --target_label $i --seed $seed --target_task $task --num_iter 1 --L 1.5 --epoch_num 4 --N_lowerBound 100 --debug_num 10000 --vAvg 20 --N_target 400 --rerun 0 --sparsity 150 --K 50 --is_linear 1    
        done
    done
done
