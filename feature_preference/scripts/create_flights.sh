# per reward
for reward in reward10
do
    for seed in 1 2 3
    do
    python3 generate_flight_data.py --env flights --config $reward --num_comparisons 1 --seed $seed &
    python3 generate_flight_data.py --env flights --config $reward --num_comparisons 3 --seed $seed &
    python3 generate_flight_data.py --env flights --config $reward --num_comparisons 5 --seed $seed &
    python3 generate_flight_data.py --env flights --config $reward --num_comparisons 10 --seed $seed &
    done
    python3 generate_flight_data.py --env flights --config $reward --num_comparisons 50 --test True &
done