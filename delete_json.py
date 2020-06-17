import os

trainPath = '/home/ubuntu/shanghai_bank_poc/tf2/train_1215'

json_files = os.listdir(trainPath)
for json_file in sorted(json_files):
    if json_file[-4:] == 'json':
        json_path = os.path.join(trainPath, json_file)
        os.remove(json_path)
