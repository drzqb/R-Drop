import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_PRF():
    with open("modelfiles/tta_rdrop_n/history.txt", "r", encoding="utf-8") as fr:
        history_n = fr.read()
        history_n = eval(history_n)

    with open("modelfiles/tta_rdrop_y/history.txt", "r", encoding="utf-8") as fr:
        history_y = fr.read()
        history_y = eval(history_y)

    gs = gridspec.GridSpec(2, 5)
    plt.subplot(gs[0, 0])
    plt.plot(history_n["loss"])
    plt.plot(history_y["loss"])
    plt.grid()
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 0])
    plt.plot(history_n["val_loss"])
    plt.plot(history_y["val_loss"])
    plt.grid()
    plt.title('val_loss')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 1])
    plt.plot(history_n["acc"])
    plt.plot(history_y["acc"])
    plt.grid()
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 1])
    plt.plot(history_n["val_acc"])
    plt.plot(history_y["val_acc"])
    plt.grid()
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 2])
    plt.plot(history_n["precision"])
    plt.plot(history_y["precision"])
    plt.grid()
    plt.title('precision')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 2])
    plt.plot(history_n["val_precision"])
    plt.plot(history_y["val_precision"])
    plt.grid()
    plt.title('val_precision')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 3])
    plt.plot(history_n["recall"])
    plt.plot(history_y["recall"])
    plt.grid()
    plt.title('recall')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 3])
    plt.plot(history_n["val_recall"])
    plt.plot(history_y["val_recall"])
    plt.grid()
    plt.title('val_recall')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 4])
    plt.plot(history_n["f1"])
    plt.plot(history_y["f1"])
    plt.grid()
    plt.title('f1')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 4])
    plt.plot(history_n["val_f1"])
    plt.plot(history_y["val_f1"])
    plt.grid()
    plt.title('val_f1')
    plt.xlabel('Epoch')
    plt.legend(['without RDrop', 'with RDrop'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("rdrop_PRF.jpg", dpi=1000, bbox_inches="tight")


if __name__ == "__main__":
    visualize_PRF()
