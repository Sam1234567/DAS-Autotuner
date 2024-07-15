
##### DES

#D = 2 kappa = 0.5

for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 2 3 0.1 0 1 2.0 0.5 0.0 0.0 0.1 40000000
done
# kappa = 1.0
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 2 3 0.1 0 1 2.0 1.0 0.0 0.0 0.1 40000000
done



#D = 4 kappa = 0.5
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 4 3 0.1 0 1 2.0 0.5 0.0 0.0 0.1 40000000
done
# kappa = 1.0
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 4 3 0.1 0 1 2.0 1.0 0.0 0.0 0.1 40000000
done


#D = 8 kappa = 0.5
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 8 3 0.1 0 1 2.0 0.5 0.0 0.0 0.1 40000000
done
# kappa = 1.0
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 8 3 0.1 0 1 2.0 1.0 0.0 0.0 0.1 40000000
done


#D = 20 kappa = 0.5
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 20 3 0.1 0 1 2.0 0.5 0.0 0.0 0.1 40000000
done
# kappa = 1.0
for i in {1..5}
do
	python3 convergence_test_ellipsoid.py 20 3 0.1 0 1 2.0 1.0 0.0 0.0 0.1 40000000
done






#figure 10, left   path  figs/dimension_scaling/sigma_0.100000_w_init_2.000000_func_3_kappa_0.5/fig2.png 
python3 dimension_scaling.py 0.5


#figure 10, right   path  figs/dimension_scaling/sigma_0.100000_w_init_2.000000_func_3/fig2.png 
python3 dimension_scaling.py 1.0



