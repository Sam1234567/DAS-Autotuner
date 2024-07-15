# Bash Script for tuning CIM with N = 300

# Collect results for DES
for i in {1..5}
do
	python3 Ising_test_ellipsoid.py 4 300 0.005 0.5 50000 5.0 2.0
done



# Collect results for SPSA
for i in {1..5}
do
	python3 Ising_test_noisyopt.py 4 300 0.005 0.1 50000 5.0 2.0 SPSA
done


# Collect results for BOHB
for i in {1..5}
do
	python3 Ising_test_bohb.py 4 300 0.005 50000 5.0 2.0
done

# Collect results for Gasnikov
for i in {1..5}
do
	python3 Ising_test_gasnikov.py 4 300 0.005 0.5 50000 5.0 2.0
done

# Figure 12, right path = figs/Ising_tune_comparison/D_4_w_init_2.000000_dt_0.500000_nsamp_max_50000/N_150_T_148.413159_beta_0.010000/fig2.png
python3 Ising_tune_comparison.py 4 300 0.005 0.5 50000 5.0 2.0


# Figure 12, left path = figs/CIM_param_plot/D_4_w_init_2.000000_dt_0.500000_nsamp_max_50000_elps/N_150_T_148.413159_beta_0.010000/fig0.png
python3 CIM_param_plot.py 4 300 0.005 0.5 50000 5.0 2.0