import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

finding_to_label = { '3401': 0, '3405': 1, '3429': 2, '3505': 3, '3576': 4, '3596': 5,
                        '3628': 6, '3634': 7, '3677': 8, '3689': 9, '3698': 10, '3713': 11,
                        '3721': 12, '3724': 13, '3732': 14, '3740': 15, '3758': 16, '3812': 17,
                        '3813': 18, '3814': 19, '3855': 20, '3871': 21, '3889': 22,
                        '3893': 23, '3898': 24, '3905': 25, '3912': 26, '3914': 27, '3916': 28,
                        '3918': 29, '3931': 30, '3941': 31, '3942': 32, '3943': 33, '3947': 34,
                        '3953': 35, '3955': 36, '3970': 37, '7836': 38, '7842': 39, '7858': 40,
                        '7895': 41  }

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]