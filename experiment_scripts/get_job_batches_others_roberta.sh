module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate activate /scratch/wh629/nlu/env

PROJECT=/scratch/wh629/bds/project
export BDS_DATA_DIR=${PROJECT}/data
export BDS_RESULTS_DIR=${PROJECT}/results
NETID=wh629
TRIALS=10
DATA=reviews_UIC-other-data
MODEL=roberta-large # 'bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large', 'albert-base-v2', 'albert-xxlarge-v2', 'albert-base-v1', 'albert-xxlarge-v1'
LENGTH=512
CAPACITY=2
PATIENCE=5
EARLY=acc # loss, acc, or f1
CHECK=1000
LOG=100
CONTENT=reviewContent,reviewCount

python hyper_parameter_tuning.py \
	--user ${NETID} \
	--n-trials ${TRIALS} \
	--dataset ${DATA} \
	--model ${MODEL} \
	--gpu-capacity ${CAPACITY} \
	--max_length ${LENGTH} \
	--patience ${PATIENCE} \
	--early_check ${EARLY} \
	--check_int ${CHECK} \
	--log_int ${LOG} \
	--accumulate \
	--content ${CONTENT} \
	--additional