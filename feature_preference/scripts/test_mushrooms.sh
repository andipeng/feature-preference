(
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_1 > ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_3 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_5 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_10 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_15 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_20 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_30 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_50 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward1 --test_network train_100 >> ../results/sim_mushrooms/reward1/0results_rlhf.txt

python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_1 > ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_3 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_5 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_10 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_15 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_20 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_30 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_50 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward1 --test_network train_100 >> ../results/sim_mushrooms/reward1/0results_featureprefs.txt
python3 ../utils/parse_results.py --in_file reward1
)&

(
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_1 > ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_3 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_5 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_10 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_15 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_20 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_30 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_50 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward2 --test_network train_100 >> ../results/sim_mushrooms/reward2/0results_rlhf.txt

python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_1 > ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_3 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_5 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_10 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_15 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_20 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_30 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_50 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward2 --test_network train_100 >> ../results/sim_mushrooms/reward2/0results_featureprefs.txt
python3 ../utils/parse_results.py --in_file reward2
)&

(
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_1 > ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_3 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_5 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_10 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_15 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_20 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_30 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_50 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward3 --test_network train_100 >> ../results/sim_mushrooms/reward3/0results_rlhf.txt

python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_1 > ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_3 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_5 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_10 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_15 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_20 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_30 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_50 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward3 --test_network train_100 >> ../results/sim_mushrooms/reward3/0results_featureprefs.txt
python3 ../utils/parse_results.py --in_file reward3
)&

(
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_1 > ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_3 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_5 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_10 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_15 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_20 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_30 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_50 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward4 --test_network train_100 >> ../results/sim_mushrooms/reward4/0results_rlhf.txt

python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_1 > ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_3 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_5 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_10 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_15 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_20 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_30 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_50 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward4 --test_network train_100 >> ../results/sim_mushrooms/reward4/0results_featureprefs.txt
python3 ../utils/parse_results.py --in_file reward4
)&

(
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_1 > ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_3 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_5 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_10 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_15 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_20 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_30 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_50 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward5 --test_network train_100 >> ../results/sim_mushrooms/reward5/0results_rlhf.txt

python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_1 > ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_3 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_5 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_10 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_15 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_20 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_30 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_50 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward5 --test_network train_100 >> ../results/sim_mushrooms/reward5/0results_featureprefs.txt
python3 ../utils/parse_results.py --in_file reward5
)&

(
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_1 > ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_3 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_5 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_10 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_15 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_20 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_30 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_50 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt
python3 test_reward.py --prefs_type rlhf --reward reward6 --test_network train_100 >> ../results/sim_mushrooms/reward6/0results_rlhf.txt

python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_1 > ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_3 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_5 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_10 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_15 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_20 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_30 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_50 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 test_reward.py --prefs_type feature_prefs --reward reward6 --test_network train_100 >> ../results/sim_mushrooms/reward6/0results_featureprefs.txt
python3 ../utils/parse_results.py --in_file reward6
)