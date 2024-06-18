"""
fait la correspondance entre les id et les race

"""
import pandas as pd
data_labels = pd.read_csv('./dog-breed-identification/labels.csv') 

target_labels = data_labels['breed']
print(len(set(target_labels))) # 120 races de chiens
print(data_labels.head())


"""
donne la race des 5 premiers chiens
"""
train_folder = './dog-breed-identification/train/'
data_labels['image_path'] = data_labels.apply(lambda row: (train_folder + row["id"] + ".jpg" ), 
                                              axis=1)
print(data_labels.head())