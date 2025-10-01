activation="sigmoid"
cpu=8
a=5
        time mpirun -n 2 --oversubscribe python3 src/colony_cants.py --data_dir ./ --data_files sample_data.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow  Main_Flm_Int --output_names Main_Flm_Int --living_time 5 --out_dir OUT_single_cants --log_dir LOG_single_cants --term_log_level INFO --log_file_name "cants_trial_ants"$a"_"$fname"_"999 --file_log_level INFO --col_log_level INFO --num_ants $a --use_cants --loss_fun mse --act_fun $activation --use_bp --bp_epochs 3
    exit



fname="withoutBP"
activation="sigmoid"
cpu=8
for i in {1..10}; do
    #for a in 5 10 15 25 35 50; do
    for a in 35; do
        echo "Starting CANTS Experiment: " $i "with $a ants"
        time mpirun -n $cpu --oversubscribe python3 src/colony_cants.py --data_dir ./ --data_files sample_data.csv --input_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow  Main_Flm_Int --output_names Main_Flm_Int --living_time 1000 --out_dir OUT_single_cants --log_dir LOG_single_cants --term_log_level INFO --log_file_name "cants_trial_ants"$a"_"$fname"_"$i --file_log_level INFO --col_log_level INFO --num_ants $a --use_cants --loss_fun mse --act_fun $activation
    exit
    done
done
