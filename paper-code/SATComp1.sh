# Bash Script for tuning CIM with N = 150



# Collect results for DES
for i in {1..5}
do
	python3 SAT_test_ensemble_ellipsoid.py 4 150 600 3 2.0 50000 5.0 0.5
done

# Collect results for DBS
for i in {1..5}
do
	python3 SAT_test_ensemble.py 4 150 600 3 2.0 50000 5.0 0.5
done

# Collect results for SPSA
for i in {1..5}
do
	python3 SAT_test_ensemble_noisyopt.py 4 150 600 3 2.0 50000 5.0 1.0 SPSA
done


# Collect results for BOHB
for i in {1..5}
do
	python3 SAT_test_ensemble_bohb.py 4 150 600 3 50000 5.0 1.0
done

# Collect results for Gasnikov
for i in {1..5}
do
	python3 SAT_test_ensemble_gasnikov.py 4 150 600 3 2.0 50000 5.0 0.5
done


#Figrue 3, left   path = figs/SAT_tune_comparison/D_4_w_init_0.500000_dt_0.500000_nsamp_max_50000/N_150_M_600_K_3_T_148.413159/fig2.png
python3 SAT_tune_comparison.py 4 150 0.01 2.0 50000 5.0 0.5

#Figure 11, left, path = 
python3 SAT_param_plot.py 4 150 0.01 2.0 50000 5.0 0.5 gas
#Figure 11, middle, path = 
python3 SAT_param_plot.py 4 150 0.01 2.0 50000 5.0 0.5 ball
#Figure 11, right, path = 
python3 SAT_param_plot.py 4 150 0.01 2.0 50000 5.0 0.5 ellips
