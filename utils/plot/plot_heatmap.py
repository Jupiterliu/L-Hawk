import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1])) #, labels=col_labels)
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0])) #, labels=row_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_heatmap(heatmap_values, strength_values, scale_values,
                 save_path="samples", x_label="Scale", y_label="Strength"):
    np.savetxt(f"{save_path}_y-axis.txt", strength_values, fmt="%s")
    np.savetxt(f"{save_path}_x-axis.txt", scale_values, fmt="%s")
    np.savetxt(f"{save_path}_heatmap_values.txt", heatmap_values, fmt='%f')

    fig, ax = plt.subplots()
    im, cbar = heatmap(heatmap_values, strength_values, scale_values,
                       ax=ax, cmap="YlGn", cbarlabel="Attack/Detection Success Rate (%)")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    # fig.tight_layout()
    # plt.show()
    # sns.set()
    # fig, ax = plt.subplots()
    # sns.heatmap(heatmap_values, ax=ax, annot=True, fmt=".1f",
    #             cmap=plt.get_cmap('Greens'),
    #             xticklabels=scale_values,
    #             yticklabels=strength_values)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title('ASR for Strength and Scale')
    plt.show()

    fig.savefig(f'{save_path}.png')

# row = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# # row = list(map(str, row))
#
# col = [100, 150, 200, 250, 300, 350, 400, 450, 500]
# # col = list(map(str, col))
#
# values = np.random.rand(len(col), len(row))
# plot_heatmap(values, col, row, "D:\\PycharmPrograms\\TPAE\\exp\\train_20231226-011149_HA_YOLOV5\\Eval_Results")
# #
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(values, row, col, ax=ax,
#                    cmap="YlGn", cbarlabel="ASR")
# texts = annotate_heatmap(im, valfmt="{x:.1f}")
# fig.tight_layout()
# plt.show()
# s = np.loadtxt(f"samples")

# str_ = np.loadtxt("D:\\PycharmPrograms\\TPAE\\exp\\train_20231226-011149_HA_YOLOV5\\Eval_Results\\strength_values.txt")
# scale_ = np.loadtxt("D:\\PycharmPrograms\\TPAE\\exp\\train_20231226-011149_HA_YOLOV5\\Eval_Results\\scale_values.txt")
# heat_ = np.loadtxt("D:\\PycharmPrograms\\TPAE\\exp\\train_20231226-011149_HA_YOLOV5\\Eval_Results\\heatmap_values.txt")
#
# plot_heatmap(heat_, str_, scale_, "D:\\PycharmPrograms\\TPAE\\exp\\train_20231226-011149_HA_YOLOV5\\Eval_Results")
# np.savetxt('example.txt', original_list)
# loaded_array = np.loadtxt('example.txt')
# loaded_list = loaded_array.tolist()
# print("Original List:", original_list)
# print("Loaded List:", loaded_list)
# print("Lists are equal:", original_list == loaded_list)