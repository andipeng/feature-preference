# per reward
for reward in reward1 reward2 reward3 reward4 reward5 reward6
do
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 1 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 3 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 5 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 10 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 15 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 20 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 30 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 50 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 100 --seed 1 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 1 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 3 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 5 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 10 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 15 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 20 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 30 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 50 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 100 --seed 2 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 1 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 3 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 5 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 10 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 15 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 20 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 30 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 50 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 100 --seed 3 &
    python3 generate_data.py --env sim_mushrooms --config $reward --num_comparisons 50 --test True &
done