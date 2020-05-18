from mcmc import run_mcmc
from lib import criss_cross, pringle, squeeze, triangle_flip, pent_to_tri
import sys

args = sys.argv
print(args)
if len(args) < 2:
    print("No test specified!")
    exit(1)

test_num      = int(args[1])
num_samples   = int(args[2])
num_landmarks = int(args[3])
num_nus       = int(args[4])
log_dir       = str(args[5])

if test_num == 0:
    run_mcmc(*criss_cross(num_landmarks=num_landmarks), num_samples, num_nus, log_dir)
elif test_num == 1:
    run_mcmc(*squeeze(num_landmarks=num_landmarks), num_samples, num_nus, log_dir)
elif test_num == 2:
    run_mcmc(*triangle_flip(num_landmarks=num_landmarks), num_samples, num_nus, log_dir)
elif test_num == 3:
    run_mcmc(*pringle(num_landmarks=num_landmarks), num_samples, num_nus, log_dir)
elif test_num == 4:
    run_mcmc(*pent_to_tri(num_landmarks=num_landmarks), num_samples, num_nus, log_dir)
else:
    print("No test specified!")
    exit(1)
