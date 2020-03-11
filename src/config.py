import os

src_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.join(src_dir, os.pardir)
data_dir = os.path.join(project_dir, "data")
data_path = os.path.join(data_dir, "beatsdataset.csv")

batch_size = 16
input_size = 71
n_classes = 23