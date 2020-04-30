module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate wsc

PROJECT=/scratch/wh629/bds/project
export BDS_DATA_DIR=${PROJECT}/data
export BDS_RESULTS_DIR=${PROJECT}/results
NETID=wh629
TRIALS=1
DATA=reviews_UIC
MODEL=roberta-large
LENGTH=512
CAPACITY=2

python hyper_parameter_tuning.py \
	--user ${NETID} \
	--n-trials ${TRIALS} \
	--dataset ${DATA} \
	--model ${MODEL} \
	--gpu-capacity ${CAPACITY} \
	--max_length ${LENGTH} \
	--accumulate