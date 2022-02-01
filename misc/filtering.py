import pandas as pd
import os 


PATH_DIR = 'FSD50K.dev_audio'
df = pd.read_csv('idx_dev.csv', names = ['ids','labels'])
ids = df['ids'].to_list()

ids_dataset = []
for file in os.listdir(PATH_DIR):
    filename = str(file)
    id_file = filename.split(".")[0]
    ids_dataset.append(int(id_file))

intersection_set = list(set.intersection(set(ids_dataset), set(ids)))

filenames = []
for ids in intersection_set:
    filename = str(ids) + ".wav"
    filenames.append(filename)

for file in os.listdir(PATH_DIR):
    filename = str(file)
    if filename not in filenames:
        os.remove('FSD50K.dev_audio/' + filename)

len(os.listdir(PATH_DIR)) # It should be equal to 5259 which is the numbe rof unique values of the dev set