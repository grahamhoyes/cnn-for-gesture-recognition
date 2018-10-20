'''
    Visualize some basic statistics of our dataset.
'''
import os
import numpy as np
import random
import matplotlib.pyplot as plt

def load_gesture(c):
    ''' Return an mx100x6 array for each gesture'''
    gesture = []
    for student in os.listdir('data'):
        if not student.startswith('student'):
            continue
        for filename in os.listdir(os.path.join('data', student)):
            if filename.split('_')[0] == c:
                data = np.loadtxt(os.path.join('data', student, filename), delimiter=',')[:, 1:] # Exclude time data
                gesture.append(data)
    #
    #             # Get an average of the data over time
    #             gesture_avg = data.mean(axis=0).tolist()
    #             gesture.append(gesture_avg)
    #
    # # Average over instances
    # res = np.array(gesture).mean(axis=0).tolist()
    # return res
    return gesture

def average_gesture(g):
    # Average over time
    time_avg = []
    for i in range(len(g)):
        time_avg.append(g[i].mean(axis=0).tolist())

    # Average over instances
    avg = np.array(time_avg).mean(axis=0).tolist()
    return avg

def stdev_gesture(g):
    # Standard deviation over time
    time_stdev = []
    for i in range(len(g)):
        time_stdev.append(g[i].std(axis=0).tolist())

    # Standard deviation over instances
    stdev = np.array(time_stdev).std(axis=0).tolist()
    return stdev

def plot():
    alphabet = [chr(i) for i in range(97, 97 + 26)]

    tick_label = [r"$a_x$", r"$a_y$", r"$a_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    fig = plt.figure(figsize=(6.1*3, 6), dpi=80)
    plt.suptitle('Average and standard deviation')
    for (i, c) in enumerate(random.sample(alphabet, 3)):  # Pick three random classes
        ax = fig.add_subplot(1, 3, i+1)
        avg = average_gesture(load_gesture(c))
        stdev = stdev_gesture(load_gesture(c))

        ax.bar(range(1, 7), avg, yerr=stdev, tick_label=tick_label, error_kw={"elinewidth": 1, "ecolor": 'r', "capsize": 4})
        ax.grid()
        ax.set_title("%s" % c)

    # plt.show()
    plt.savefig('figs/bar.png')

if __name__ == "__main__":
    plot()