for reward in reward10
do
    for seed in 1 2 3
    do
        python3 train_reward.py --prefs_type rlhf --env flights --reward $reward --data_file train_1 --epochs 2000 --batch_size 1 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --env flights --reward $reward --data_file train_3 --epochs 2000 --batch_size 3 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --env flights --reward $reward --data_file train_5 --epochs 2000 --batch_size 5 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --env flights --reward $reward --data_file train_10 --epochs 2000 --batch_size 10 --seed $seed &
        wait

        python3 train_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --data_file train_1 --epochs 4000 --batch_size 1 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --data_file train_3 --epochs 4000 --batch_size 2 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --data_file train_5 --epochs 4000 --batch_size 5 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --data_file train_10 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        wait
        
        python3 train_reward.py --prefs_type feature_prefs_gt --env flights --reward $reward --data_file train_1 --epochs 4000 --batch_size 1 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_gt --env flights --reward $reward --data_file train_3 --epochs 4000 --batch_size 2 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_gt --env flights --reward $reward --data_file train_5 --epochs 4000 --batch_size 5 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_gt --env flights --reward $reward --data_file train_10 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &

    done
done