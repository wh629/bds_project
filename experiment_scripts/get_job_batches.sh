module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate wsc

export NLU_RESULTS_DIR=/scratch/wh629/nlu/projects/wsc/results
export NLU_DATA_DIR=/scratch/wh629/nlu/projects/wsc/data
NETID=wh629
TRIALS=50
DATA=wsc-spacy
FRAME=P-SPAN
CAPACITY=1

python hyper_parameter_tuning_wh.py \
	--user ${NETID} \
	--n-trials ${TRIALS} \
	--dataset ${DATA} \
	--framing ${FRAME} \
	--gpu-capacity ${CAPACITY}