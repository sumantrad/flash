# flash
flash type attention kernel tuning on A4000

# clone
git clone ...

## env. setup
uv .venv --python <version>

source .venv/bin/activate

uv pip install -r requirements.txt

## profile kernels
make executable:
chmod +x ncu_profile.sh

(update block dim., num_warps, num_stages warmup etc. on file and change file and output names in script)
(requires NSight installation; check https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
(update ncu path accordingly in script)

run:
./ncu_profile.sh

## best version: Kernel_5

Tuning is done on A4000. To tune on your system -

1. please run 'python kernel_5_attention1_tune.py' to get the best configuration for attention kernel 1
2. please run 'python kernel_5_attention2_tune.py' to get the best configuration for attention_kernel_2
3. now profile them jointly by using 'kernel_5.py' (./ncu_profile.sh)
4. accuracy check using 'kernel_5_accuracy.py'


