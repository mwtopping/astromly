import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Denoise_Model(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()


        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.enc3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.enc4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.enc5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.enc6 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # concat 96 params

        self.deconv5ab = nn.Sequential(
            nn.ConvTranspose2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))

        #concat to 144 params

        self.deconv4ab = nn.Sequential(
            nn.ConvTranspose2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.deconv3ab = nn.Sequential(
            nn.ConvTranspose2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.deconv2ab = nn.Sequential(
            nn.ConvTranspose2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.deconv1ab = nn.Sequential(
            nn.ConvTranspose2d(96+in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

    def forward(self, x):

        #print("Input", x.shape)
        pool1 = self.enc1(x)
        #print("Pool1", pool1.shape)
        pool2 = self.enc2(pool1)
        #print("Pool2", pool2.shape)
        pool3 = self.enc3(pool2)
        #print("Pool3", pool3.shape)
        pool4 = self.enc4(pool3)
        #print("Pool4", pool4.shape)
        pool5 = self.enc5(pool4)
        #print("Pool5", pool5.shape)
        conv6 = self.enc6(pool5)
        #print("conv6", conv6.shape)

        
        concat5 = torch.cat((conv6, pool4), dim=1)
        upsample4 = self.deconv5ab(concat5)
        #print("upsample4", upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.deconv4ab(concat4)
        #print("upsample3", upsample4.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.deconv3ab(concat3)
        #print("upsample2", upsample4.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.deconv2ab(concat2)
        #print("upsample1", upsample4.shape)
        concat1 = torch.cat((upsample1, x), dim=1)
        output = self.deconv1ab(concat1)

        return output



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")




def get_model():
    device = get_device()

    NN = Denoise_Model()
    NN.to(device)
    return NN, device
#        # create training data
#        BATCH_SIZE=1
#        dataset = myDataset(device, N=10000)
#        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#
#        # create some more data for testing
#        testdataset = myDataset(device)
#        testdataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#
#        loss_fn = nn.MSELoss()
#        optimizer = torch.optim.Adam(NN.parameters(), lr=1e-4)
#        train(NN, dataloader, testdataloader, 10, loss_fn, optimizer, device=device)
#
#        # save model
#        if savemodel:
#            torch.save(NN.state_dict(), "./models/model.pth")
#
#
#    
#    return NN, device

if __name__ == "__main__":

    NN, device = get_model()

    test_data = torch.rand((1, 1, 128, 128), device=device)
    NN.eval()
    result = NN(test_data)
    print(result.shape)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(test_data[0,0,:,:].cpu().detach().numpy())
    ax[1].imshow(result[0,0,:,:].cpu().detach().numpy())
    plt.show()

