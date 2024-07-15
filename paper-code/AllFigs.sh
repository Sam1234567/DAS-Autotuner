#this shell script renders most of the important figures once the appropriate data files have been generated


#figure 2, left   path  figs/benchmark_plot/D_2_fidx_12/fig3.png
python3 benchmark_plot.py 2 12 0 0.05
#table 3 latex outputted

#figure 2, middle   path   figs/benchmark_plot/D_4_fidx_12/fig3.png
python3 benchmark_plot.py 4 12 0 0.05
#table 1 latex outputted

#figure 2, right   path   figs/benchmark_plot/D_8_fidx_15/fig1.png
python3 benchmark_plot.py 8 15 1 0.025
#table 4 latex outputted



#Figrue 3, path = figs/SAT_tune_comparison/D_4_w_init_0.500000_dt_0.500000_nsamp_max_50000/N_150_M_600_K_3_T_148.413159/fig2.png
python3 SAT_tune_comparison.py 4 150 0.01 2.0 50000 5.0 0.5

#Figure 11 left, path = 
python3 SAT_param_plot.py 4 150 0.01 2.0 50000 5.0 0.5 gas
#Figure 11 middle, path = 
python3 SAT_param_plot.py 4 150 0.01 2.0 50000 5.0 0.5 ball
#Figure 11 right, path = 
python3 SAT_param_plot.py 4 150 0.01 2.0 50000 5.0 0.5 ellips


#figure 10, left   path  figs/dimension_scaling/sigma_0.100000_w_init_2.000000_func_3_kappa_0.5/fig2.png 
python3 dimension_scaling.py 0.5


#figure 10, right   path  figs/dimension_scaling/sigma_0.100000_w_init_2.000000_func_3/fig2.png 
python3 dimension_scaling.py 1.0


#Figure 12, left and figure 3, right  path = figs/Ising_tune_comparison/D_4_w_init_2.000000_dt_0.500000_nsamp_max_50000/N_150_T_148.413159_beta_0.010000/fig2.png
python3 Ising_tune_comparison.py 4 150 0.01 0.5 50000 5.0 2.0


#Figure 13, left   path = figs/CIM_param_plot/D_4_w_init_2.000000_dt_0.500000_nsamp_max_50000_elps/N_150_T_148.413159_beta_0.010000/fig0.png
python3 CIM_param_plot.py 4 150 0.01 0.5 50000 5.0 2.0


#Figure 12, right  path = figs/Ising_tune_comparison/D_4_w_init_2.000000_dt_0.500000_nsamp_max_50000/N_150_T_148.413159_beta_0.010000/fig2.png
python3 Ising_tune_comparison.py 4 300 0.005 0.5 50000 5.0 2.0


#Figure 13, right  path = figs/CIM_param_plot/D_4_w_init_2.000000_dt_0.500000_nsamp_max_50000_elps/N_150_T_148.413159_beta_0.010000/fig0.png
python3 CIM_param_plot.py 4 300 0.005 0.5 50000 5.0 2.0

