import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np
import torch
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)

from IPython import display
from sklearn.metrics import euclidean_distances


def plot_print_rewards_stats(rewards_all_episodes,
                             show_every,
                             args,
                             mode="show",
                             save_path=""):
    # unpack the args
    seq = args[0]
    seed = args[1]
    num_episodes = args[2]

    # Calculate and print the average reward per show_every episodes
    rewards_per_N_episodes = np.split(
        np.array(rewards_all_episodes),
        num_episodes
    )
    count = show_every

    # for plotting
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    print("\n********Stats per {} episodes********\n".format(show_every))
    for r in rewards_per_N_episodes:
        # print(count, "avg: ", str(sum(r/show_every)))
        # print(count, "min: ", str(min(r)))
        # print(count, "max: ", str(max(r)))

        aggr_ep_rewards['ep'].append(count)
        aggr_ep_rewards['avg'].append(sum(r / show_every))
        aggr_ep_rewards['min'].append(min(r))
        aggr_ep_rewards['max'].append(max(r))

        count += show_every

    # Width, height in inches.
    # default: [6.4, 4.8]
    fig_width = 6.4
    fig_height = 4.8
    # adjust the height of the histogram
    if np.array(rewards_all_episodes).max() - np.array(rewards_all_episodes).min() > 10:
        fig_width = 6.5
        fig_height = 6.5
    fig, subplot = plt.subplots(figsize=(fig_width, fig_height))
    # Be sure to only pick integer tick locations
    subplot.yaxis.set_major_locator(MaxNLocator(integer=True))
    # subplot.yaxis.set_major_locator(MultipleLocator(1))
    subplot.yaxis.set_minor_locator(MultipleLocator(1))

    subplot.set_xlabel('Episode Index')
    subplot.set_ylabel('Episode Reward')

    subplot.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    subplot.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    subplot.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")

    # Put a legend below current axis
    subplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=True, ncol=3)

    # split the seq into chunks of 10 for the matplotlib title
    chunks, chunk_size = len(seq), 10
    seq_title_list = [
        seq[i:i + chunk_size] + "\n" for i in range(0, chunks, chunk_size)
    ]
    seq_title_str = ''.join(seq_title_list)

    # print("Title: ", title)

    plt.grid(True, which="major", lw=1.2, linestyle='-')
    plt.grid(True, which="minor", lw=0.8, linestyle='--')
    plt.tight_layout()
    if mode == "show":
        plt.show()
    elif mode == "save":
        # save the pdf fig with seq name
        plt.savefig("{}Seq_{}-{}-Eps{}-Seed{}.png".format(
            save_path,  # "./xxx"
            seq,
            num_episodes,
            seed,
        ))
    plt.close()


def plot_2D_foleded_protein(labelled_conf):
    """
    input:
        labelled_conf:
            transformed file sequence of xy coords with state:
            ((x,y), 'H|P')
            e.g:
            [((0, 0), 'H'),
            ((0, 1), 'H'),...
            ((3, 1), 'P')]
    output:
        plot.show
    """

    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)


    fig = plt.figure()
    subplot = fig.add_subplot()
    subplot.title.set_text("Folded Protein")
    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]

    H_points = [t[0] for t in labelled_conf if t[1] == 'H']
    P_points = [t[0] for t in labelled_conf if t[1] == 'P']

    max_xval = np.max(x)
    max_yval = np.max(y)

    total_max = max(max_xval, max_yval)
    min_xval = np.min(x)
    min_yval = np.min(y)

    total_min = min(min_xval, min_yval)
    subplot.set_xlim(total_min - 0.1, total_max+0.1)
    subplot.set_ylim(total_min - 0.1, total_max+0.1 )

    subplot.set_aspect('equal')  # , adjustable='box')

    subplot.xaxis.set_major_locator(ticker.MultipleLocator(1))
    subplot.yaxis.set_major_locator(ticker.MultipleLocator(1))

    subplot.plot(
        x, y,
        color='cornflowerblue',
        linewidth=4,
        label="backbone",
    )

    subplot.plot(
        [h[0] for h in H_points],
        [h[1] for h in H_points],
        'o',
        color='royalblue',
        markersize=14,
        label="H",
    )

    for index in range(0, len(labelled_conf)):
        for jndex in range(index, len(labelled_conf)):
            if abs(index - jndex) >= 2:
                current_amino_acid_i = labelled_conf[index][1]
                current_amino_acid_j = labelled_conf[jndex][1]
                current_place_i = labelled_conf[index][0]
                current_place_j = labelled_conf[jndex][0]
                x_i = current_place_i[0]
                y_i = current_place_i[1]
                x_j = current_place_j[0]
                y_j = current_place_j[1]
                if current_amino_acid_i == 'H' and current_amino_acid_j == 'H' and (
                        abs(x_i - x_j) + abs(y_i - y_j) == 1):
                    subplot.plot([x_i, x_j], [y_i, y_j], '--', color='mediumblue')
                    subplot.plot([x_i, x_j], [y_i, y_j], 'o', color='mediumblue', markersize=14,)

    subplot.plot(
        [p[0] for p in P_points],
        [p[1] for p in P_points],
        'o',
        color='orange',
        fillstyle='none',
        markersize=14,
        label="P",
    )

    plt.show()


def save_2d_protein_to_xyz(labelled_conf, output_file_name):
    output_file = open(output_file_name, "w")
    for atom_item in labelled_conf:
        coords, atom = atom_item
        x, y = coords
        z = 0
        output_file.write("{}\t {}\t {}\t {}\t\n".format(atom, x, y, z))
    output_file.close()


def plot_3D_foleded_protein(labelled_conf):
    """
    input:
        labelled_conf:
            transformed file sequence of xy coords with state:
            ((x,y), 'H|P')
            e.g:
            [((0, 0), 'H'),
            ((0, 1), 'H'),...
            ((3, 1), 'P')]
    output:
        plot.show
    """


    plt.rc('axes', labelsize=25)
    plt.rc('xtick', labelsize=21)
    plt.rc('ytick', labelsize=21)
    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]
    z = [t[0][2] for t in labelled_conf]

    H_points = [t[0] for t in labelled_conf if t[1] == 'H']
    P_points = [t[0] for t in labelled_conf if t[1] == 'P']

    fig = plt.figure()
    subplot = fig.add_subplot(projection='3d')
    subplot.title.set_text("Folded Protein")

    max_xval = np.max(x)
    max_yval = np.max(y)
    max_zval = np.max(z)
    total_max = max(max_xval, max_yval, max_zval)
    min_xval = np.min(x)
    min_yval = np.min(y)
    min_zval = np.min(z)
    total_min = min(min_xval, min_yval, min_zval)
    subplot.set_xlim(total_min, total_max)
    subplot.set_ylim(total_min, total_max)
    subplot.set_zlim(total_min, total_max)


    subplot.grid(linewidth=0.6, linestyle=':')


    subplot.set_aspect('equal')  # , adjustable='box')


    subplot.xaxis.set_major_locator(ticker.MultipleLocator(1))
    subplot.yaxis.set_major_locator(ticker.MultipleLocator(1))
    subplot.zaxis.set_major_locator(ticker.MultipleLocator(1))

    subplot.set_xlabel("x coord")
    subplot.set_ylabel("y coord")
    subplot.set_zlabel("z coord")


    subplot.plot(
        x, y, z,
        color='cornflowerblue',
        linewidth=4,
        label="backbone",
    )


    subplot.plot(
        [h[0] for h in H_points],
        [h[1] for h in H_points],
        [h[2] for h in H_points],
        'o',
        color='royalblue',
        markersize=14,
        label="H",
    )

    subplot.plot(
        [p[0] for p in P_points],
        [p[1] for p in P_points],
        [p[2] for p in P_points],
        'o',
        color='orange',
        fillstyle='none',
        markersize=14,
        label="P",
    )

    for index in range(0, len(labelled_conf)):
        for jndex in range(index, len(labelled_conf)):
            if abs(index - jndex) >= 2:
                current_amino_acid_i = labelled_conf[index][1]
                current_amino_acid_j = labelled_conf[jndex][1]
                current_place_i = labelled_conf[index][0]
                current_place_j = labelled_conf[jndex][0]
                x_i = current_place_i[0]
                y_i = current_place_i[1]
                z_i = current_place_i[2]
                x_j = current_place_j[0]
                y_j = current_place_j[1]
                z_j = current_place_j[2]
                if current_amino_acid_i == 'H' and current_amino_acid_j == 'H' and (
                        abs(x_i - x_j) + abs(y_i - y_j) + abs(z_i - z_j) == 1):
                    subplot.plot([x_i, x_j], [y_i, y_j],[z_i,z_j], '--', color='mediumblue')
                    subplot.plot([x_i, x_j], [y_i, y_j], [z_i,z_j], 'o', color='mediumblue', markersize=14, )

    plt.show()


def plot_moving_avg(scores, n=2, mode="show", save_path=""):
    print("means = ", scores.mean())

    # useful utility function for graphing the average
    def moving_average(a, n=n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(scores, n=n))
    # plt.plot(moving_average(opt_scores, n=500))
    # plt.plot(moving_average(rand_scores, n=500))
    if mode == "show":
        plt.show()
    elif mode == "save":
        # save the pdf fig with seq name
        plt.savefig(save_path + "moving_avg-" + str(n) + ".png")
    plt.close()


def plot_loss(episodes, losses):
    # unpack the args

    # Width, height in inches.
    # default: [6.4, 4.8]

    fig, subplot = plt.subplots(
    )

    # subplot.yaxis.set_major_locator(MultipleLocator(1))


    subplot.set_xlabel('Episode Index')
    subplot.set_ylabel('Episode Reward')

    subplot.plot(episodes,losses, label="Losses")

    plt.show()
    plt.close()