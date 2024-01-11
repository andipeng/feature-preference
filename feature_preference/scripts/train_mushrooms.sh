for reward in reward1 reward2 reward3 reward4 reward5 reward6
do
    for seed in 1 2 3
    do
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_1 --epochs 2000 --batch_size 1 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_3 --epochs 2000 --batch_size 3 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_5 --epochs 2000 --batch_size 5 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_10 --epochs 2000 --batch_size 10 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_15 --epochs 2000 --batch_size 10 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_20 --epochs 2000 --batch_size 10 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_30 --epochs 2000 --batch_size 10 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_50 --epochs 2000 --batch_size 20 --seed $seed &
        python3 train_reward.py --prefs_type rlhf --reward $reward --data_file train_100 --epochs 2000 --batch_size 50 --seed $seed &
        wait

        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_1 --epochs 4000 --batch_size 1 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_3 --epochs 4000 --batch_size 3 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_5 --epochs 4000 --batch_size 5 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_10 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_15 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_20 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_30 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_50 --epochs 4000 --batch_size 20 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs --reward $reward --data_file train_100 --epochs 4000 --batch_size 50 --seed $seed --alpha 0.5 --beta 0.5 &
        wait

        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_1 --epochs 4000 --batch_size 1 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_3 --epochs 4000 --batch_size 2 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_5 --epochs 4000 --batch_size 5 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_10 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_15 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_20 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_30 --epochs 4000 --batch_size 10 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_50 --epochs 4000 --batch_size 20 --seed $seed --alpha 0.5 --beta 0.5 &
        python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_100 --epochs 4000 --batch_size 50 --seed $seed --alpha 0.5 --beta 0.5 &
        wait

        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_1 --epochs 4000 --batch_size 1 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_3 --epochs 4000 --batch_size 2 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_5 --epochs 4000 --batch_size 5 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_10 --epochs 4000 --batch_size 10 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_15 --epochs 4000 --batch_size 10 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_20 --epochs 4000 --batch_size 10 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_30 --epochs 4000 --batch_size 10 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_50 --epochs 4000 --batch_size 20 --seed $seed --alpha 1. --beta 0. &
        # python3 train_reward.py --prefs_type feature_prefs_human --reward $reward --data_file train_100 --epochs 4000 --batch_size 50 --seed $seed --alpha 1. --beta 0. &
    done
done