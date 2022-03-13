import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import glob
import random
from PIL import Image
import os

# set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "will be used.")
writer = SummaryWriter('runs')
step = 0

input_shape = 48
input_size = input_shape * input_shape
train_seq_length = 100


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
    def __init__(self, root, randomizer=1, randomize=True):
        self.trajectories = sorted(glob.glob(f'{root}/*/*'))
        self.randomizer = randomizer
        self.randomize = randomize

    def __len__(self):
        return len(self.trajectories)*self.randomizer

    def __getitem__(self, index):
        index = index % len(self.trajectories)
        traj_dir, frames = self.trajectories[index], []
        for f in sorted(glob.glob(f'{traj_dir}/*.npy')):
            frame_data = np.load(f)
            frame = Image.fromarray(np.uint8(frame_data*255))
            frame = frame.resize((input_shape, input_shape))
            frame = transforms.ToTensor()(frame).view(-1)
            frames.append(frame)
        frames = torch.stack(frames)
        start_ind = 0 if self.randomize else random.randint(0, int(train_seq_length*0.15))
        coalesce = os.path.basename(os.path.dirname(traj_dir)) == "coalesce"
        return frames[start_ind:start_ind+train_seq_length, :], frames[start_ind+train_seq_length:, :], torch.tensor(coalesce).type(torch.float32)


train_ds = CustomDataset("data/train")
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
test_ds = CustomDataset("data/test", randomize=False)
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


class CoalescenceClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(CoalescenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        context_size = num_layers * hidden_size * 2
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(context_size, 1), nn.Sigmoid())

    def forward(self, hidden, cell):
        assert hidden.shape == cell.shape and hidden.shape[0] == self.num_layers and hidden.shape[-1] == self.hidden_size, print(hidden.shape, cell.shape)
        hidden = hidden.squeeze(1).view(1, -1)
        cell = cell.squeeze(1).view(1, -1)
        rnn_context_vec = torch.stack((hidden, cell), dim=1)
        return self.fc(rnn_context_vec).squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
    
    def forward(self, source, target_seq_len, target=None, epoch=None):
        # source shape: [BS, seq_len, NXN]
        assert source.shape[1] == train_seq_length and source.shape[2] == input_size, print(source.shape)
        bs = source.shape[0]
        outputs = torch.zeros(bs, target_seq_len, input_size).to(device)

        hidden, cell = self.encoder(source[:, :-1, :])
        coa_cla_prob = self.classifier(hidden, cell)
        x = source[:, -1, :]

        teacher_force_ratio = 0.
        if target is not None:
            teacher_force_ratio = 0.8 - 0.02 * epoch  # use target to learn in earlier epochs and gradually start self learning
        
        for t in range(target_seq_len):
            output, hidden, cell = self.decoder(x.unsqueeze(1), hidden, cell)
            outputs[:, t, :] = output
            if random.random() >= teacher_force_ratio:
                x = output
            else:
                x = target[:, t, :]
        return outputs, coa_cla_prob


def evaluate_testMSE(model, dl, device, epoch):
    acc_cri = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        traj_errors, cla_errors = [], []
        for ti, batch in enumerate(dl):
            inp, tar, coa = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output, coalescence_prob = model(inp.to(device), tar.shape[1])
            if ti < 10:
                print(f'Case has {tar.shape[1]} frames')
                save_gif(tar.cpu().detach().squeeze(0), f'results/test_target_{ti}_epoch{epoch}')
                print(f'Predicted {output.shape[1]} frames')
                save_gif(output.cpu().detach().squeeze(0), f'results/test_output_{ti}_epoch{epoch}')
            traj_errors.append(acc_cri(output, tar).item())
            cla_errors.append(torch.round(coalescence_prob) != coa)
        avg_traj_error = sum(traj_errors)/len(traj_errors)
        avg_cla_error = sum(cla_errors)/len(cla_errors)
        writer.add_scalar('Test avg. MSE error', avg_traj_error, global_step=epoch)
        writer.add_scalar('Test avg. CLA error', avg_cla_error, global_step=epoch)
    return avg_traj_error, avg_cla_error


hidden_size = 2048
num_layers = 3
dropouts = 0.4

encoder_net = Encoder(input_size, hidden_size, num_layers, dropouts)
decoder_net = Decoder(input_size, hidden_size, num_layers, dropouts)
coalescence_classifier = CoalescenceClassifier(hidden_size, num_layers)
model = Seq2Seq(encoder_net, decoder_net, coalescence_classifier)
# ckpt = torch.load("models/model_20.tar")
# model.load_state_dict(ckpt['state_dict'])
model = model.to(device)

lr = 5e-3
traj_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
opt_traj = torch.optim.Adam(traj_params, lr=lr)
opt_cla = torch.optim.Adam(model.classifier.parameters(), lr=lr)
num_epochs = 20
loss_cri = nn.BCELoss()

for epoch in range(num_epochs):
    print("epoch", epoch)
    for bi, batch in enumerate(train_dl):
        model.train()
        inp, tar, coa = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # train for shorter sequences first
        target_seq_length = 100 + 30*epoch
        tar = tar[:, :target_seq_length, :]
        output, coalescence_prob = model(inp, tar.shape[1], target=tar, epoch=epoch)
        assert output.shape == tar.shape, print(output.shape, tar.shape)

        opt_cla.zero_grad()
        loss_cla = loss_cri(coalescence_prob, coa)
        loss_cla.backward(retain_graph=True)
        opt_cla.step()
        opt_traj.zero_grad()
        loss_traj = loss_cri(output, tar)
        loss_traj.backward()
        nn.utils.clip_grad_norm_(traj_params, max_norm=1)
        opt_traj.step()

        writer.add_scalar('Trajectory training loss', loss_traj, global_step=step)
        writer.add_scalar('Coalescence classification loss', loss_cla, global_step=step)
        step += 1
        if bi % 50 == 0:
            print(f'Case has {tar.shape[1]} frames')
            save_gif(tar.cpu().detach().squeeze(0), f'results/train_target_{bi}_epoch{epoch}')
            print(f'Predicted {output.shape[1]} frames')
            save_gif(output.cpu().detach().squeeze(0), f'results/train_output_{bi}_epoch{epoch}')

    if epoch > 4 and epoch % 3 == 1:
        evaluate_testMSE(model, test_dl, device, epoch)
    model.train()
    torch.save({'state_dict': model.state_dict()}, f'models/model_{epoch+1}.tar')

# reload model after training
ckpt = torch.load(f'models/model_{num_epochs}.tar', map_location=device)
model.load_state_dict(ckpt['state_dict'])
test_traj_error, test_cla_error = evaluate_testMSE(model, test_dl, device, num_epochs)
print("Trained model's average MSE for trajectory generation is:", test_traj_error)
print("Trained model's average classification accuracy for coalescence prediction is:", 1-test_cla_error)
