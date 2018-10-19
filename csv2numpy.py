'''
    Save the data in the .csv file, save as a .npy file in ./data
'''

import os
import numpy as np

def load_csv():
    instances = []
    labels = []

    for student in os.listdir('data'):
        if os.path.isfile(os.path.join('data', student)):
            continue
        print("Student %s" % student)
        for fname in os.listdir(os.path.join('data', student)):
            label = fname.split('_')[0]
            data = np.loadtxt(os.path.join('data', student, fname), delimiter=',')
            instance = data[:, 1:] # Remove the first time column
            #instance_row = instance.reshape(instance.size).tolist() # Make a vector
            # Let's make this 3d instead of 2d
            # Each index is still a new instance, just easier to access the columns
            instances.append(instance.tolist())
            labels.append(label)

    np.save('data/instances.npy', np.array(instances))
    np.save('data/labels.npy', np.array(labels))

if __name__ == "__main__":
    load_csv()