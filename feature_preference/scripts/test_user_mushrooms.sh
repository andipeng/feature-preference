for reward in reward6
do
    for seed in 1 2 3 4 5
    do
        (
        python3 test_reward.py --prefs_type rlhf --env user_study --reward $reward --test_network train_1 --seed $seed > ../results/user_study/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env user_study --reward $reward --test_network train_3 --seed $seed >> ../results/user_study/$reward/$seed/0results_rlhf.txt
        python3 test_reward.py --prefs_type rlhf --env user_study --reward $reward --test_network train_5 --seed $seed >> ../results/user_study/$reward/$seed/0results_rlhf.txt

        python3 test_reward.py --prefs_type feature_prefs --env user_study --reward $reward --test_network train_1 --seed $seed > ../results/user_study/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env user_study --reward $reward --test_network train_3 --seed $seed >> ../results/user_study/$reward/$seed/0results_featureprefs.txt
        python3 test_reward.py --prefs_type feature_prefs --env user_study --reward $reward --test_network train_5 --seed $seed >> ../results/user_study/$reward/$seed/0results_featureprefs.txt

        python3 test_reward.py --prefs_type feature_prefs_human --env user_study --reward $reward --test_network train_1 --seed $seed > ../results/user_study/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env user_study --reward $reward --test_network train_3 --seed $seed >> ../results/user_study/$reward/$seed/0results_featureprefshuman.txt
        python3 test_reward.py --prefs_type feature_prefs_human --env user_study --reward $reward --test_network train_5 --seed $seed >> ../results/user_study/$reward/$seed/0results_featureprefshuman.txt

        python3 ../utils/parse_seed_results.py --env user_study --reward $reward --seed $seed
        )
    done
    wait
    python3 ../utils/parse_results.py --env user_study --reward $reward
done