from os import wait
import torch
from astropy.visualization import ZScaleInterval
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import get_device, ImageDataset
from model import get_model


from torch.utils.data import DataLoader


def median_binner(a,bin_x,bin_y):
    m,n = np.shape(a)
    strided_reshape = np.lib.stride_tricks.as_strided(a,shape=(bin_x,bin_y,m//bin_x,n//bin_y),strides = a.itemsize*np.array([(m // bin_x) * n, (n // bin_y), n, 1]))
    return np.array([np.median(col) for row in strided_reshape for col in row]).reshape(bin_x,bin_y)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calc_loss(inp_batch, targ_batch, model, device):
    inp_batch.to(device)
    targ_batch.to(device)


def evalutae_model(model, train_loader, test_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, test_loss



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.mse_loss(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss

    return total_loss / num_batches



def train(model, dataloader, testdataloader, Nepochs, loss_fn, optimizer, device=torch.device("cpu")):
    train_losses = []
    test_losses = []
    steps = []
    global_step = -1
    eval_freq = 20
    for ii in tqdm(range(Nepochs)):
        total_loss = 0
        model.train()
        for inp_batch, targ_batch in dataloader:
            optimizer.zero_grad()

            # calc the loss
            output = model(inp_batch)
            loss = loss_fn(output, targ_batch)
            loss.backward()
            optimizer.step()
            if global_step % eval_freq == 0:
                train_loss, test_loss = evalutae_model(model, dataloader, testdataloader, device, 10)
    
                train_losses.append(train_loss.cpu())
                test_losses.append(test_loss.cpu())
                steps.append(global_step)

            global_step += 1

    plt.figure()
    plt.plot(steps, train_losses)
    plt.plot(steps, test_losses)

if __name__ == "__main__":


    scaler = ZScaleInterval()

    device = get_device()
    data = ImageDataset(device)
    testdata = ImageDataset(device)


    test_img = data.fullimages[1]
    print(test_img)

    dataloader = DataLoader(data, batch_size=16, shuffle=True)
    testdataloader = DataLoader(testdata, batch_size=16, shuffle=True)

#    data_batch, labels_batch = next(iter(dataloader))
#    print(f"Batch shape: {data_batch.shape}")
#    print(f"Labels shape: {labels_batch.shape}")



    print(f"Creating model on device:{device}")
    model, device = get_model()
    #model.compile()

    print(f"Instantiated model with {count_parameters(model)} parameters.")



    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, weight_decay=1e-5)



    train(model, dataloader, testdataloader, 30, loss_fn, optimizer, device=device)


    model.eval()
    data_batch, labels_batch = next(iter(dataloader))


    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    test_img = data.raw_images[1][:2048, :2048]

    limits = scaler.get_limits(test_img)
    ax[0].imshow(test_img, vmin=limits[0], vmax=limits[1])

    print(test_img)
    print(np.shape(test_img))
    inp_tensor = torch.tensor(test_img).unsqueeze(0).unsqueeze(0).to(device)
    print(inp_tensor.shape)
    res = model(inp_tensor)

    limits = scaler.get_limits(res[0,0,:,:].cpu().detach().numpy())
    ax[1].imshow(res[0,0,:,:].cpu().detach().numpy(), vmin=limits[0], vmax=limits[1])
    print(res)

    for d in data_batch:
        print(d)
        print(d.shape)
        res = model(d.unsqueeze(0))
        print(res)
        fig, axs = plt.subplots(1, 3)#, sharex=True, sharey=True)

        limits = scaler.get_limits(d[0,:,:].cpu().detach().numpy())
        axs[0].imshow(d[0,:,:].cpu().detach().numpy(), vmin=limits[0], vmax=limits[1])
        axs[1].imshow(res[0,0,:,:].cpu().detach().numpy(), vmin=limits[0], vmax=limits[1])
        outimg = res[0,0,:,:].cpu().detach().numpy()

        out_binned = median_binner(outimg, 64, 64)
        axs[2].imshow(out_binned, vmin=limits[0], vmax=limits[1])
        plt.show()


    plt.show()
