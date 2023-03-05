import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)
from sklearn.metrics import euclidean_distances


def plot_print_rewards_stats(rewards_all_episodes,
                             show_every,
                             args,
                             mode="show",
                             save_path=""):
    # unpack the args
    seq = args.seq
    seed = args.seed
    num_episodes = args.num_episodes

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
        aggr_ep_rewards['avg'].append(sum(r/show_every))
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Be sure to only pick integer tick locations
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel('Episode Index')
    ax.set_ylabel('Episode Reward')

    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

    # split the seq into chunks of 10 for the matplotlib title
    chunks, chunk_size = len(seq), 10
    seq_title_list = [
        seq[i:i+chunk_size]+"\n" for i in range(0, chunks, chunk_size)
    ]
    seq_title_str = ''.join(seq_title_list)
    title = "{}Algo={}, Epi={}, Seed={}\nShow-every {}".format(
        seq_title_str,
        algo,
        num_episodes,
        seed,
        show_every,
    )
    # print("Title: ", title)
    plt.title(title)
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
            algo,
            num_episodes,
            seed,
        ))
    plt.close()



def plot_HPSandbox_conf_2D(labelled_conf,
                        display_mode="draw",
                        pause_t=0.5,
                        save_fig=False,
                        save_path="",
                        score=2022,
                        optima_idx=0,
                        info={}):
    """
    input:
        labelled_conf:
            transformed file sequence of xy coords with state:
            ((x,y), 'H|P')
            e.g:
            [((0, 0), 'H'),
            ((0, 1), 'H'),...
            ((3, 1), 'P')]
        display_mode:
            draw vs show
        pause_t:
            seconds to display draw in plt.pause()
        save_fig:
            whether to save the pdf fig with seq name
            otherwise leaves a fig-live.pdf for review
    output:
        plot.show
    """


    plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=21)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=21)    # fontsize of the tick labels

    # print("+=+=+=+=+=+ plot_HPSandbox_conf -_-_-_-_-_-")
    fig = plt.figure()
    ax = fig.add_subplot()

    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]

    str_seq = ''.join([t[1] for t in labelled_conf])
    assert len(str_seq) == info["chain_length"]
    H_seq = [t[0] for t in labelled_conf if t[1] == 'H']
    P_seq = [t[0] for t in labelled_conf if t[1] == 'P']


    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # the HP plot consists of three layers

    # layer 1: backbone with solid line
    ax.plot(
        x, y,
        color='cornflowerblue',
        linewidth=4,
        label="backbone",
    )
    # layer 2: H as solid blue dots
    ax.plot(
        [h[0] for h in H_seq],
        [h[1] for h in H_seq],

        'o',
        markersize=14,
        label="H",
    )
    # layer 3: P as hollow orange dots
    ax.plot(
        [p[0] for p in P_seq],
        [p[1] for p in P_seq],
        'o',
        fillstyle='none',
        markersize=14,
        label="P",
    )

    plt.show()


def plot_HPSandbox_conf_3D(labelled_conf,
                        display_mode="draw",
                        pause_t=0.5,
                        save_fig=False,
                        save_path="",
                        score=2022,
                        optima_idx=0,
                        info={}):
    """
    input:
        labelled_conf:
            transformed file sequence of xy coords with state:
            ((x,y), 'H|P')
            e.g:
            [((0, 0), 'H'),
            ((0, 1), 'H'),...
            ((3, 1), 'P')]
        display_mode:
            draw vs show
        pause_t:
            seconds to display draw in plt.pause()
        save_fig:
            whether to save the pdf fig with seq name
            otherwise leaves a fig-live.pdf for review
    output:
        plot.show
    """


    plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=21)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=21)    # fontsize of the tick labels

    # print("+=+=+=+=+=+ plot_HPSandbox_conf -_-_-_-_-_-")
    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]
    z = [t[0][2] for t in labelled_conf]
    str_seq = ''.join([t[1] for t in labelled_conf])
    assert len(str_seq) == info["chain_length"]
    H_seq = [t[0] for t in labelled_conf if t[1] == 'H']
    P_seq = [t[0] for t in labelled_conf if t[1] == 'P']


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # set x and y limit and center origin
    max_xval = np.max(x)
    max_yval = np.max(y)
    max_zval = np.max(z)
    total_max = max(max_xval,max_yval, max_zval)
    min_xval = np.min(x)
    min_yval = np.min(y)
    min_zval = np.min(z)
    total_min = min(min_xval, min_yval, min_zval)
    ax.set_xlim(total_min, total_max)
    ax.set_ylim(total_min, total_max)
    ax.set_zlim(total_min, total_max)

    # grid background
    ax.grid(linewidth=0.6, linestyle=':')

    # adjust plots with equal axis ratios
    #ax.axis('equal')
    ax.set_aspect('equal')  # , adjustable='box')

    # x and y axis tick at integer level
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))


    # axis title
    ax.set_xlabel("x coord")
    ax.set_ylabel("y coord")
    ax.set_zlabel("z coord")

    # the HP plot consists of three layers

    # layer 1: backbone with solid line
    ax.plot(
        x, y,z,
        color='cornflowerblue',
        linewidth=4,
        label="backbone",
    )
    # layer 2: H as solid blue dots
    ax.plot(
        [h[0] for h in H_seq],
        [h[1] for h in H_seq],
        [h[2] for h in H_seq],
        'o',
        markersize=14,
        label="H",
    )
    # layer 3: P as hollow orange dots
    ax.plot(
        [p[0] for p in P_seq],
        [p[1] for p in P_seq],
        [p[2] for p in P_seq],
        'o',
        fillstyle='none',
        markersize=14,
        label="P",
    )

    plt.show()
