# per reward
for reward in reward1 reward2 reward3 reward4 reward5 reward6
do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
    python3 parse_user_data.py --env user_study --config $reward --num_comparisons 1 --seed $seed &
    python3 parse_user_data.py --env user_study --config $reward --num_comparisons 3 --seed $seed &
    python3 parse_user_data.py --env user_study --config $reward --num_comparisons 5 --seed $seed &
    done
done