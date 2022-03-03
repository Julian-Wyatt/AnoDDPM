def main():
    pass


def load_csv(volume: str, arg_num):
    metrics_df = pd.read_csv(f"./metrics/ARGS={arg_num}/{volume}.csv")
    metrics_transpose = metrics_df.T
    metrics_transpose.reset_index(level=[None], inplace=True)
    new_col_names = ["timestep", "Dice Coefficient", "SSIM", "IOU", "Precision", "Recall", "False Positive Rate"]
    metrics_transpose.rename(
            {metrics_transpose.columns[i]: new_col_names[i] for i in range(len(new_col_names))}, axis=1,
            inplace=True
            )
    # Convert index to column

    metrics_transpose['timestep'] = metrics_transpose['timestep'].astype(int)
    return metrics_transpose


def load_ROC_csv(volume: str):
    with open(f"./metrics/ROC_data/{volume}.csv", mode="r") as f:
        data = f.readlines()
    return data


def graph_dice():
    simplex_ex1 = load_csv("19691", 28)
    gauss_ex1 = load_csv("19691", 26)

    simplex_ex2 = load_csv("18756", 28)
    gauss_ex2 = load_csv("18756", 26)

    x_axis = np.arange(min(simplex_ex1['timestep']), max(simplex_ex1['timestep']), 100)
    x_axis += [1000]

    for i in ["Dice Coefficient", "IOU"]:
        plt.plot(simplex_ex1['timestep'], simplex_ex1[i], "-", label="Patient 19691 - Simplex")
        plt.plot(gauss_ex1['timestep'], gauss_ex1[i], "-", label="Patient 19691 - Gaussian")
        plt.plot(simplex_ex1['timestep'], simplex_ex2[i], "--", label="Patient 18756 - Simplex")
        plt.plot(gauss_ex1['timestep'], gauss_ex2[i], "--", label="Patient 18756 - Gaussian")
        plt.legend()
        plt.ylabel(i)
        plt.xlabel("Timestep $t$")
        ax = plt.gca()
        # ax.set_xticks(x_axis)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1000])
        plt.savefig(f"./paper_images/{i} Graph.png")
        plt.show()
        plt.clf()

    conv = lambda x: [float(i) for i in x.split(",")]
    ROC = load_ROC_csv("18756")
    plt.plot(conv(ROC[0]), conv(ROC[1]), "-", label="Simplex")
    plt.plot(conv(ROC[3]), conv(ROC[4]), ":", label="Gaussian")
    plt.plot(conv(ROC[6]), conv(ROC[7]), "-.", label="Adversarial Context Encoder")
    plt.legend()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    ax = plt.gca()
    # ax.set_xticks(x_axis)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import pandas as pd
    import numpy as np

    font_path = "./times new roman.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams['figure.dpi'] = 1000

    graph_dice()
