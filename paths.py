import os

# Where the repository exists
base_path = "/h/stephaneao/sentence_encoders/"
bert_config_file = os.path.join(base_path, "bert_config.json")

# Where you want to save models (this requires lots of space - better on hhds)
save_path = "/scratch/hdd001/home/stephaneao/"
pretrained_path = os.path.join(save_path, "pretrained_berts")
finetuned_path = os.path.join(save_path, "finetuned_berts")

# Where you are loading the data from (better on ssd if possible for faster reads)
data_path = "/scratch/ssd001/home/stephaneao"
glue_data_path = os.path.join(data_path, "glue_data")
train_data_path = os.path.join(data_path, "train_data")
