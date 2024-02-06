import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as st
import numpy as np 

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Encoder(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), 
            nn.init.calculate_gain('relu'))
        
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=1, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        self.out_size = 1024
        dim = 2048 * 4
        self.fc = nn.Sequential(
            init_(nn.Linear(dim + self.num_actions, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, self.out_size)),
            nn.ReLU(),
        )


    def forward(self, panos, actions):
        if panos.dim() == 3:
            panos = panos.unsqueeze(0)
        B, H, W, C = panos.shape 
        panos = panos.reshape((B*4, -1, W, C))
        panos = panos.permute(0, 3, 1, 2)
        features = self.feat_extract(panos)
        features = features.reshape((B, -1))
        concat = torch.cat((features, actions.reshape((B, -1))), dim=-1)
        mean, log_var = self.fc(concat).chunk(2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_shape, decoder_layer_sizes, in_dim, noise_dim, num_labels, no_noise):
        super().__init__()
        self.output_shape = output_shape
        self.no_noise = no_noise
        if self.no_noise:
            noise_dim = 0
        model = [nn.Linear(in_dim+noise_dim, decoder_layer_sizes[0]),
                 nn.ReLU()]
        for i in range(len(decoder_layer_sizes)-1):
            model.append(nn.Linear(decoder_layer_sizes[i], decoder_layer_sizes[i+1]))
            model.append(nn.ReLU())
        model.append(nn.Linear(decoder_layer_sizes[-1], np.prod(self.output_shape)*num_labels))
        model.append(nn.Sigmoid())
        self.net = nn.Sequential(*model)
        self.num_labels = num_labels

    def forward(self, z, noise): 
        if self.no_noise:
            concat = z
        else:
            concat = torch.cat((z, noise), dim=-1)
        x = self.net(concat)
        W, H = self.output_shape
        x = x.view(-1, W, H, self.num_labels) 
        return x

class PredModel(nn.Module):
    def __init__(self, decoder_layer_sizes, 
                       noise_layer_sizes,
                       num_actions,
                       noise_dim, 
                       no_noise,
                       num_labels):
        super().__init__()
        input_shape = (7, 7)
        self.encoder = Encoder((7, 7, 3), num_actions)
        self.no_noise = no_noise
        out_size = self.encoder.out_size // 2
        self.decoder1 = Decoder(input_shape, decoder_layer_sizes, out_size, noise_dim, num_labels[0], no_noise)
        self.decoder2 = Decoder(input_shape, decoder_layer_sizes, out_size, noise_dim, num_labels[1], no_noise)
        self.decoder3 = Decoder(input_shape, decoder_layer_sizes, out_size, noise_dim, num_labels[2], no_noise)
        if not self.no_noise:
            self.noise = NoisePred(out_size, noise_layer_sizes, noise_dim)
        self.num_labels = num_labels

    def forward(self, panos, actions):
        mean, log_var = self.encoder(panos.float(), actions.float())
        z = self.reparameterize(mean, log_var)
        if self.no_noise:
            pred_obj, pred_color, pred_cond = self.decoder1(z, None), self.decoder2(z, None), self.decoder3(z, None)
        else:
            noise_mu, noise_log_var = self.noise(z)
            rand_noise = self.reparameterize(noise_mu, noise_log_var)
            pred_obj, pred_color, pred_cond = self.decoder1(z, rand_noise), self.decoder2(z, rand_noise), self.decoder3(z, rand_noise)
        
        return [pred_obj, pred_color, pred_cond], mean, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_single(self, pred, y, mean, log_var):
        B = pred.shape[0]
        pred, y = pred.float(), y.float()
        pred, y = pred.view(B,-1), y.view(B,-1)
        BCE = torch.nn.functional.binary_cross_entropy(pred, y, reduction='sum')/B
        return BCE

    def loss(self, preds, ys, mean, log_var):
        loss1 = self.loss_single(preds[0], ys[0], mean, log_var)
        loss2 = self.loss_single(preds[1], ys[1], mean, log_var)
        loss3 = self.loss_single(preds[2], ys[2], mean, log_var)
        return loss1 + loss2 + loss3

class NoisePred(nn.Module):
    def __init__(self, encoding_dim, layer_sizes, latent_dim):
        super().__init__()
        self.input_shape = encoding_dim
        self.latent_dim = latent_dim
        model = [nn.Linear(encoding_dim, layer_sizes[0]),
                 nn.ReLU()]
        for i in range(len(layer_sizes)-1):
            model.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            model.append(nn.ReLU())
        model = model[:-1]
        model.append(nn.Tanh())
        model.append(nn.Linear(layer_sizes[-1], latent_dim*2))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(-1, self.input_shape).detach()
        mean, log_var = self.net(x).chunk(2, dim=1)
        return mean, log_var

    def loss(self, mem_enc, gt_enc):
        mu_mem, log_std_mem = self.forward(mem_enc)
        p = torch.distributions.Normal(mu_mem, log_std_mem.exp())

        mu, log_std = self.forward(gt_enc)
        q = torch.distributions.Normal(mu, log_std.exp())
        kl_loss = torch.distributions.kl_divergence(p, q).mean()
        return kl_loss