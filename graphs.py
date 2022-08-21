import evaluation


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
    with open(f"./metrics/ROC_data_2/{volume}.csv", mode="r") as f:
        data = f.readlines()
    return data


def conv_csv_2_mu_std(volume: str, arg_num, mean_distance=20):
    metrics_df = pd.read_csv(f"./metrics/ARGS={arg_num}/{volume}.csv")
    print(metrics_df.columns)
    for i in ["Dice", "IOU", "SSIM", "Precision", "Recall", "FPR"]:

        metrics_df[f"{i}_std"] = metrics_df[f"{i}"].rolling(mean_distance).std()
        metrics_df[f"{i}_mu"] = metrics_df[f"{i}"].rolling(mean_distance).mean()
        metrics_df[f"{i}_std"] = metrics_df[f"{i}_std"].fillna(method='backfill')
        metrics_df[f"{i}_mu"] = metrics_df[f"{i}_mu"].fillna(method='backfill')
        metrics_df[f"{i}_error_min"] = metrics_df[f"{i}_mu"] - metrics_df[f"{i}_std"]
        metrics_df[f"{i}_error_max"] = metrics_df[f"{i}_mu"] + metrics_df[f"{i}_std"]

        plt.plot(metrics_df['timestep'], metrics_df[f"{i}_mu"], "-", label="mu")
        plt.fill_between(
                metrics_df['timestep'], metrics_df[f"{i}_error_min"], metrics_df[f"{i}_error_max"], alpha=0.5
                )
        plt.legend()
        plt.ylabel(f"{i}")
        plt.xlabel("Timestep $t$")
        plt.title(f"{volume},{arg_num}")
        ax = plt.gca()
        # ax.set_xticks(x_axis)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1000])
        # plt.savefig(f"./paper_images/{i} Graph.png")
        plt.show()
        plt.clf()

    metrics_df.to_csv(
            f"./paper_images/ARGS={arg_num}-{volume}-precision-recall.csv", mode="w",
            index=False, columns=["timestep", "Precision_mu", "Precision_error_min", "Precision_error_max",
                                  "Recall_mu", "Recall_error_min", "Recall_error_max"]
            )


def make_ROC_csv():
    # evaluation.ROC_AUC()
    conv = lambda x: np.array([float(i) for i in x.split(",")])
    csv = load_ROC_csv("Square_Errors")
    """
    csv = [simplex_sqe, gauss_sqe, GAN_sqe, img_256, img_128]
    """
    simplex_sqe = conv(csv[0])
    gauss_sqe = conv(csv[1])
    GAN_sqe = conv(csv[2])
    img_256 = conv(csv[3])
    img_128 = conv(csv[4])

    fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC(img_256, simplex_sqe)
    fpr_gauss, tpr_gauss, _ = evaluation.ROC_AUC(img_256, gauss_sqe)
    fpr_GAN, tpr_GAN, _ = evaluation.ROC_AUC(img_128, GAN_sqe)
    print(len(fpr_GAN), len(fpr_gauss), len(fpr_simplex))

    for model in [(fpr_simplex, tpr_simplex, "simplex"), (fpr_gauss, tpr_gauss, "gauss"), (fpr_GAN, tpr_GAN, "GAN")]:

        with open(f'./paper_images/temp-{model[2]}.csv', mode="w") as f:
            f.write(f"fpr, tpr, {evaluation.AUC_score(model[0], model[1])}")
            f.write("\n")
            for i in range(len(model[0])):
                f.write(",".join([f"{j:.4f}" for j in [model[0][i], model[1][i]]]))
                f.write("\n")

    plt.plot(fpr_simplex, tpr_simplex, "-", label="Simplex")
    # plt.plot(conv(ROC[3]), conv(ROC[4]), ":", label="Gaussian")
    # plt.plot(conv(ROC[6]), conv(ROC[7]), "-.", label="Adversarial Context Encoder")
    plt.legend()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    ax = plt.gca()
    # ax.set_xticks(x_axis)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.show()


def graph_dice():
    simplex_ex1 = load_csv("19691", 28)
    gauss_ex1 = load_csv("19691", 26)

    simplex_ex2 = load_csv("18756", 28)
    gauss_ex2 = load_csv("18756", 26)

    mu = []
    std = []
    col = simplex_ex2["Dice Coefficient"].to_list()
    for i, val in enumerate(col):
        if i >= 2:
            mu.append(np.mean(col[i - 2:i + 2]))
            std.append(np.std(col[i - 2:i + 2]))
        elif i == 1:
            mu.append(np.mean(col[i - 1:i + 3]))
            std.append(np.std(col[i - 1:i + 3]))
        elif i == 0:
            mu.append(np.mean(col[i:i + 4]))
            std.append(np.std(col[i:i + 4]))
        elif i == len(col) - 2:
            mu.append(np.mean(col[-5:-1]))
            std.append(np.std(col[-5:-1]))
        elif i == len(col) - 1:
            mu.append(np.mean(col[-4:]))
            std.append(np.std(col[-4:]))

    mu = np.array(mu)
    std = np.array(std)
    with open("temp.csv", mode="w") as f:
        f.write("timestep,mu,std\n")
        for i in range(len(mu)):
            f.write(f"{simplex_ex1['timestep'].iloc[i]},{mu[i]},{std[i]}\n")

    x_axis = np.arange(min(simplex_ex1['timestep']), max(simplex_ex1['timestep']), 100)
    x_axis += [1000]

    for i in ["Dice Coefficient"]:
        plt.plot(simplex_ex1['timestep'], simplex_ex1[i], "-", label="Patient 19691 - Simplex")
        plt.plot(gauss_ex1['timestep'], gauss_ex1[i], "-", label="Patient 19691 - Gaussian")
        plt.plot(simplex_ex1['timestep'], simplex_ex2[i], "--", label="Patient 18756 - Simplex")
        plt.plot(gauss_ex1['timestep'], gauss_ex2[i], "--", label="Patient 18756 - Gaussian")
        plt.plot(simplex_ex1['timestep'], mu)
        plt.fill_between(simplex_ex1['timestep'], mu - std, mu + std)
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


    #
    #
    # for i in ["Dice Coefficient", "IOU"]:
    #     plt.plot(simplex_ex1['timestep'], simplex_ex1[i], "-", label="Patient 19691 - Simplex")
    #     plt.plot(gauss_ex1['timestep'], gauss_ex1[i], "-", label="Patient 19691 - Gaussian")
    #     plt.plot(simplex_ex1['timestep'], simplex_ex2[i], "--", label="Patient 18756 - Simplex")
    #     plt.plot(gauss_ex1['timestep'], gauss_ex2[i], "--", label="Patient 18756 - Gaussian")
    #     plt.legend()
    #     plt.ylabel(i)
    #     plt.xlabel("Timestep $t$")
    #     ax = plt.gca()
    #     # ax.set_xticks(x_axis)
    #     ax.set_ylim([0, 1])
    #     ax.set_xlim([0, 1000])
    #     plt.savefig(f"./paper_images/{i} Graph.png")
    #     plt.show()
    #     plt.clf()

    # conv = lambda x: [float(i) for i in x.split(",")]
    # ROC = load_ROC_csv("18756")
    # plt.plot(conv(ROC[0]), conv(ROC[1]), "-", label="Simplex")
    # plt.plot(conv(ROC[3]), conv(ROC[4]), ":", label="Gaussian")
    # plt.plot(conv(ROC[6]), conv(ROC[7]), "-.", label="Adversarial Context Encoder")
    # plt.legend()
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False Positive Rate")
    # ax = plt.gca()
    # # ax.set_xticks(x_axis)
    # ax.set_ylim([0, 1])
    # ax.set_xlim([0, 1])
    # plt.show()


def reduce_quality(filename, reduce_size=5):
    out_fpr = []
    out_tpr = []
    overall_fpr = []
    overall_tpr = []
    with open(filename, mode="r") as f:
        headers = f.readline()
        newLine = f.readline()
        rolling_mean_fpr = []
        rolling_mean_tpr = []

        while newLine:

            fpr_line, tpr_line = newLine.split(",")
            rolling_mean_fpr.append(float(fpr_line))
            rolling_mean_tpr.append(float(tpr_line))
            overall_fpr.append(float(fpr_line))
            overall_tpr.append(float(tpr_line))

            if len(rolling_mean_fpr) > reduce_size:
                out_fpr.append(np.mean(rolling_mean_fpr))
                out_tpr.append(np.mean(rolling_mean_tpr))
                rolling_mean_fpr = []
                rolling_mean_tpr = []

            newLine = f.readline()
        out_fpr.append(np.mean(rolling_mean_fpr))
        out_tpr.append(np.mean(rolling_mean_tpr))
    out_tpr.append(1)
    out_fpr.append(1)
    with open(f"{filename[:-4]}_reduced_{reduce_size}.csv", mode="w") as f:
        f.write(headers)
        for fpr, tpr in zip(out_fpr, out_tpr):
            f.write(f"{fpr},{tpr}\n")

    plt.plot(out_fpr, out_tpr, label="after")
    plt.plot(overall_fpr, overall_tpr, ":", label="before")
    plt.legend()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    ax = plt.gca()
    # ax.set_xticks(x_axis)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.show()


def ROC_graph():
    df = pd.read_csv(f"./metrics/AnoGAN_ROC.csv")
    print(df.head(), repr(df.columns))
    plt.plot(df["fpr"], df[" tpr"])
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
    # plt.rcParams['figure.dpi'] = 1000

    # graph_dice()\
    # make_ROC_csv()
    # ROC_graph()
    # reduce_quality("./paper_images/temp-simplex.csv", 4)
    # reduce_quality("./paper_images/temp-GAN.csv", 4)
    # reduce_quality("./paper_images/temp-gauss.csv", 2)
    reduce_quality("./metrics/ROC_data_2/overall_simplex.csv", 1000)
    reduce_quality("./metrics/ROC_data_2/overall_gauss.csv", 1000)
    reduce_quality("./metrics/ROC_data_2/overall_hybrid.csv", 1000)
    reduce_quality("./metrics/ROC_data_2/overall_GAN.csv", 30)

    # reduce_quality("./metrics/AnoGAN_ROC.csv", 20)
    # conv_csv_2_mu_std("19691", 26)
    # conv_csv_2_mu_std("18756", 26)
    # conv_csv_2_mu_std("19691", 28)
    # conv_csv_2_mu_std("18756", 28)
