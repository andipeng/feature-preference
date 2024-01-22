# per reward
for reward in reward1 reward2 reward3 reward4 reward5 reward6
do
    for seed in 1 2 3
    do
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 1 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 3 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 5 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 10 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 15 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 20 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 30 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 50 --seed $seed &
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 100 --seed $seed &
    done
    python3 generate_mushroom_data.py --env sim_mushrooms --config $reward --num_comparisons 50 --test True &
done