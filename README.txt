To compile test_hpc_1:

mpic++ -mcmodel=medium test_hpc_1.c timer.c -o test_hpc_1
test_hpc_1.c:
timer.c:
(PHPC) [mbower@v001 PHPC_hybrid]$ mpirun ./test_hpc_1


To compile later test_hpc_X:

$ mpic++ -acc -ta=tesla:managed -Minfo=accel -fopenmp test_hpc_6.c timer.c -o test_hpc_6
test_hpc_6.c:
...
timer.c:
(PHPC) [mbower@v001 PHPC_hybrid]$ mpirun ./test_hpc_6

