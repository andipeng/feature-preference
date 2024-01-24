for reward in reward10
do
    for seed in 1 2 3
    do
        (
        python3 test_reward.py --prefs_type rlhf --env flights --reward $reward --test_network train_1 --seed $seed > ../results/flights/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env flights --reward $reward --test_network train_3 --seed $seed >> ../results/flights/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env flights --reward $reward --test_network train_5 --seed $seed >> ../results/flights/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env flights --reward $reward --test_network train_10 --seed $seed >> ../results/flights/$reward/$seed/0results_rlhf.txt

        # python3 test_reward.py --prefs_type feature_prefs --env flights --reward $reward --test_network train_1 --seed $seed > ../results/flights/$reward/$seed/0results_featureprefs.txt
        # python3 test_reward.py --prefs_type feature_prefs --env flights --reward $reward --test_network train_3 --seed $seed >> ../results/flights/$reward/$seed/0results_featureprefs.txt
        # python3 test_reward.py --prefs_type feature_prefs --env flights --reward $reward --test_network train_5 --seed $seed >> ../results/flights/$reward/$seed/0results_featureprefs.txt
        # python3 test_reward.py --prefs_type feature_prefs --env flights --reward $reward --test_network train_10 --seed $seed >> ../results/flights/$reward/$seed/0results_featureprefs.txt

        python3 test_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --test_network train_1 --seed $seed > ../results/flights/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --test_network train_3 --seed $seed >> ../results/flights/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --test_network train_5 --seed $seed >> ../results/flights/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env flights --reward $reward --test_network train_10 --seed $seed >> ../results/flights/$reward/$seed/0results_featureprefshuman.txt
        python3 ../utils/parse_seed_results.py --env flights --reward $reward --seed $seed
        )
    done
    wait
    python3 ../utils/parse_results.py --env flights --reward $reward
done