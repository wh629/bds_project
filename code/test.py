import myio
import transformers

data_handler = myio.IO(data_dir = "../../dataset/",
                       task_names=["Test"],
                       tokenizer=transformers.AutoTokenizer.from_pretrained('albert-base-v2'),

                       )

data_handler.read_task()