import torch
from torch.nn import functional as F


from torch.utils.data import DataLoader
from torch.optim import Optimizer

from datasets import AlexDataset


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                    epoch: int, device: torch.device):

    # setting the model in training mode
    model.train(True)

    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0

    for X, y in dataloader:
        # Putting images and targets on given device
        X = X.to(device)
        y = y.to(device)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(X)

        # calculating loss and backpropagation the loss through the network
        loss = F.cross_entropy(outputs, y)
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        print(f'loss: {loss.item()}, step progression: {batch}/{n_batches}, epoch: {epoch}')

        batch += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


@torch.no_grad()
def test_accuracy(model: torch.nn.Module, dataset: AlexDataset, device: torch.device):
    # set model in evaluation mode. turns of dropout layers and other layers which only are used for training. same
    # as .train(False)
    model.eval()

    # how many correct classified images
    cnt = 0

    for i in range(len(dataset)):
        # gets image an corresponding target
        X, y = dataset.__getitem__(i)

        # puts tensors onto devices
        X = X.to(device)
        y = y.to(device)

        # reshapes image to (1, C, H, W), model will only take images in batches, so here batch of one
        X = X.view(-1, 3, 227, 227)

        # pass image and get output vector
        output = model(X)

        # check argmax is same as target
        if torch.argmax(output) == torch.argmax(y):
            cnt += 1

    # number of correct predicted / total number of samples
    accuracy = cnt / len(dataset)

    return accuracy


# Testing that test_accuracy work correct
if __name__ == '__main__':
    from model import Alexnet

    model = Alexnet(2, 3)
    dataset = AlexDataset('image_data/DD_resized/test')

    acc = test_accuracy(model, dataset)

    print()
    print(acc)
