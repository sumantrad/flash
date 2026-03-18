# flash
flash type attention kernel tuning

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


