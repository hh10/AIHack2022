import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import glob
import random
from PIL import Image

# use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "will be used.")

input_shape = 48
input_size = input_shape * input_shape
train_seq_length = 100
min_test_target_seq_length = 300


def disparity_normalization(disp):  # disp is an array in uint8 data type
    _min = np.amin(disp)
    _max = np.amax(disp)
    disp_norm = (disp - _min) * 255.0 / (_max - _min)
    disp_norm = np.uint8(disp_norm)
    return disp_norm


def save_gif(framest, name):
    frames_np = framest.numpy()
    frames = []
    for fi in range(frames_np.shape[0]):
        res_arr = disparity_normalization(frames_np[fi]).reshape((input_shape, input_shape))
        frame = Image.fromarray(res_arr)
        frames.append(frame)
    frames[0].save(fp=f'{name}.gif', format='GIF', append_images=frames[1:], save_all=True, loop=0)


class CustomDataset(Dataset):
    def __init__(self, root, randomizer=1):
        self.trajectories = sorted(glob.glob(f'{root}/*/*'))
        self.randomizer = randomizer

    def __len__(self):
        return len(self.trajectories)*self.randomizer

    def __getitem__(self, index):
        index = index % self.randomizer
        frames = []
        for f in sorted(glob.glob(f'{self.trajectories[index]}/*.npy')):
            frame_data = np.load(f)
            frame = Image.fromarray(np.uint8(frame_data*255))
            frame = frame.resize((input_shape, input_shape))
            frame = transforms.ToTensor()(frame).view(-1)
            frames.append(frame)
        frames = torch.stack(frames)
        start_ind = 0 if self.randomizer == 1 else random.randint(train_seq_length*0.9)
        return frames[start_ind:start_ind+train_seq_length, :], frames[start_ind+train_seq_length:, :]
        

train_ds = CustomDataset("data/train")
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
test_ds = CustomDataset("data/test")
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Dropout = nn.Dropout(p)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p, batch_first=True)

    def forward(self, x):
        # x: [BS, seq_length, input_size]
        assert x.shape[1] == train_seq_length-1 and x.shape[2] == input_size, print(x.shape)
        _, (hidden, cell) = self.rnn(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Dropout = nn.Dropout(p)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, input_size), nn.Sigmoid())

    def forward(self, x, hidden, cell):
        # x: [BS, 1, input_size]
        assert x.shape[1] == 1 and x.shape[2] == input_size, print(x.shape)
        outputs, (hidden, cell) = self.rnn(x, (hidden, cell))
        predictions = self.fc(outputs.unsqueeze(1)).squeeze(1).squeeze(1)
        # predictions: [BS, input_size]
        assert predictions.shape[1] == input_size, print(predictions.shape)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target_seq_len, target=None, epoch=None):
        # source shape: [BS, seq_len, NXN]
        assert source.shape[1] == train_seq_length and source.shape[2] == input_size, print(source.shape)
        bs = source.shape[0]
        outputs = torch.zeros(bs, target_seq_len, input_size).to(device)

        hidden, cell = self.encoder(source[:, :-1, :])
        x = source[:, -1, :]

        teacher_force_ratio = 0.
        if target is not None:
            teacher_force_ratio = 0.9 - 0.02 * epoch  # use target to learn in earlier epochs and gradually start self learning
        
        for t in range(target_seq_len):
            output, hidden, cell = self.decoder(x.unsqueeze(1), hidden, cell)
            outputs[:, t, :] = output
            if random.random() >= teacher_force_ratio:
                x = output
            else:
                x = target[:, t, :]
        return outputs


def evaluate_testMSE(model, dl, device, epoch):
    acc_cri = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        errors = []
        for ti, tbatch in enumerate(dl):
            inp, tar = batch[0].to(device), batch[1].to(device)       
            output = model(inp.to(device), tar.shape[1])
            if ti < 3:
                print(f'Predicted for {tar.shape[1]} frames')
                save_gif(output.cpu().detach().squeeze(0), f'test_output_{ti}_epoch{epoch}')
            errors.append(acc_cri(output, tar).item())
        avg_test_errors_batch = sum(errors)/len(errors)
        writer.add_scalar('Test avg. MSE error', avg_test_errors_batch, global_step=epoch)
    return avg_test_errors_batch


num_epochs = 20
lr = 5e-3
loss_cri = nn.BCELoss(reduction='sum')

hidden_size = 2560
num_layers = 3
dropouts = 0.35

writer = SummaryWriter('runs')
step = 0

encoder_net = Encoder(input_size, hidden_size, num_layers, dropouts)
decoder_net = Decoder(input_size, hidden_size, num_layers, dropouts)
model = Seq2Seq(encoder_net, decoder_net).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

for ti, (tinp, ttar) in enumerate(test_dl):
    if ti > 2:
        break
    save_gif(tinp.squeeze(0), f'results/test_input_{ti}')
    print(f'Test case has {ttar.shape[1]} frames')
    save_gif(ttar.squeeze(0), f'results/test_target_{ti}')


for epoch in range(num_epochs):
    print("epoch", epoch)
    for bi, batch in enumerate(train_dl):
        model.train()
        inp, tar = batch[0].to(device), batch[1].to(device)
        output = model(inp, tar.shape[1], target=tar, epoch=epoch)
        # print(torch.max(output), torch.max(tar))
        assert output.shape == tar.shape, print(output.shape, tar.shape)

        # output, tar reshaping
        opt.zero_grad()
        loss = loss_cri(output, tar)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        opt.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1
        if bi % 50 == 0:
            save_gif(output.cpu().detach().squeeze(0), f'results/train_output_{bi}_epoch{epoch}')

    if epoch > 4 and epoch % 3 == 1:
        evaluate_testMSE(model, test_dl, device, epoch)
    torch.save({'state_dict': model.state_dict()}, f'models/model_{epoch+1}.tar')

# reload model after training
ckpt = torch.load(f'models/model_{num_epochs}.tar', map_location=device)
model.load_state_dict(ckpt['state_dict'])
print("Trained model's average MSE is:", evaluate_testMSE(model, test_dl, device, num_epochs))
