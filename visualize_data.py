'''
    Visualize some samples.
'''
import random
import os
import numpy as np
import matplotlib.pyplot as plt


def get_data(student, letter, instance):
    data = np.loadtxt(os.path.join('data', 'student%d' % student, '%s_%d.csv' % (letter, instance)), delimiter=',')
    return data


alphabet = [chr(i) for i in range(97, 97+26)]

for c in random.sample(alphabet, 2): # Pick two random classes
    fig = plt.figure(figsize=(6.1*3, 6), dpi=80)
    plt.suptitle('Sensor values for %s' % c)
    j = 1
    for s in random.sample(range(43), 3): # Pick three random students
        i = random.randint(1, 5) # Pick a random instance

        ax = fig.add_subplot(1, 3, j)

        # Read the file
        data = get_data(s, c, i)

        # Plot each sensor value
        ax.plot(data[:, 0], data[:, 1], label=r"$a_x$")
        ax.plot(data[:, 0], data[:, 2], label=r"$a_y$")
        ax.plot(data[:, 0], data[:, 3], label=r"$a_z$")
        ax.plot(data[:, 0], data[:, 4], label=r"$\omega_x$")
        ax.plot(data[:, 0], data[:, 5], label=r"$\omega_y$")
        ax.plot(data[:, 0], data[:, 6], label=r"$\omega_z$")

        ax.set_xlabel(r'Time ($ms$)')
        ax.set_ylabel('Raw sensor value')

        ax.grid()
        ax.legend()
        ax.set_title('student%d, %s_%d' % (s, c, i))

        j+=1

    plt.savefig('figs/visualize_data_%s.png' % c)



