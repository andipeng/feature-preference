for reward in reward1 reward2 reward3 reward4 reward5 reward6
do
    for seed in 1 2 3 4 5
    do
        (
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_1 --seed $seed > ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_3 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_5 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_10 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_15 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_20 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_30 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_50 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env sim_mushrooms --reward $reward --test_network train_100 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhf.txt

        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_1 --seed $seed > ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_3 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_5 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_10 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_15 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_20 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_30 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_50 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env sim_mushrooms --reward $reward --test_network train_100 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefs.txt

        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_1 --seed $seed > ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_3 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_5 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_10 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_15 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_20 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_30 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_50 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env sim_mushrooms --reward $reward --test_network train_100 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_featureprefshuman.txt

        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_1 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_3 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_5 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_10 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_15 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_20 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_30 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_50 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 test_reward.py --prefs_type rlhf_human --env sim_mushrooms --reward $reward --test_network train_100 --seed $seed >> ../results/sim_mushrooms/$reward/$seed/0results_rlhfhuman.txt
        python3 ../utils/parse_seed_results.py --env sim_mushrooms --reward $reward --seed $seed
        )
    done
    wait
    python3 ../utils/parse_results.py --env sim_mushrooms --reward $reward
done