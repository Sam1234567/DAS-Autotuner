
##### DES

#Collect results fro DES D = 2
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 2 12 0.1 1 0 0.25 0.5 0.05 0.0 0.5 4000000
done


#Collect results fro DES D = 4
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 4 12 0.1 1 0 0.25 0.5 0.05 0.0 0.5 4000000
done


#Collect results fro DES D = 8
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 8 15 0.1 1 0 0.25 0.5 0.025 0.0 0.5 4000000
done



##### SPSA

#Collect results fro SPSA D = 2
for i in {1..5}
do
	python3 convergence_test_noisyopt.py 2 12 0.1 1 0 2.0 40000000
done


#Collect results fro SPSA D = 4
for i in {1..5}
do
	python3 convergence_test_noisyopt.py 4 12 0.1 1 0 2.0 40000000
done


#Collect results fro SPSA D = 8
for i in {1..5}
do
	python3 convergence_test_noisyopt.py 8 15 0.1 1 0 2.0 40000000
done



##### BOHB

#Collect results fro BOHB D = 2
for i in {1..5}
do
	python3 convergence_test_bohb.py 2 12 0.1 1 0 40000000
done


#Collect results fro BOHB D = 4
for i in {1..5}
do
	python3 convergence_test_bohb.py 4 12 0.1 1 0 40000000
done


#Collect results fro BOHB D = 8
for i in {1..5}
do
	python3 convergence_test_bohb.py 8 15 0.1 1 0 40000000
done




##### Gasnikov

#Collect results fro Gasnikov D = 2
for i in {1..5}
do
	python3 convergence_test_gasnikov.py 2 12 0.1 1 0 0.25 40000000
done


#Collect results fro Gasnikov D = 4
for i in {1..5}
do
	python3 convergence_test_gasnikov.py 4 12 0.1 1 0 0.25 40000000
done


#Collect results fro Gasnikov D = 8
for i in {1..5}
do
	python3 convergence_test_gasnikov.py 8 15 0.1 1 0 0.25 40000000
done




#figure 2, left   path  figs/benchmark_plot/D_2_fidx_12/fig3.png
python3 benchmark_plot.py 2 12 0 0.05
#table 1 latex outputted

#figure 2, middle   path   figs/benchmark_plot/D_4_fidx_12/fig3.png
python3 benchmark_plot.py 4 12 0 0.05
#table 2 latex outputted

#figure 2, right   path   figs/benchmark_plot/D_8_fidx_15/fig1.png
python3 benchmark_plot.py 8 15 1 0.025
#table 3 latex outputted


