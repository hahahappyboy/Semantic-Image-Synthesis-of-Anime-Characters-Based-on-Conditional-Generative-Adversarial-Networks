import torch.nn as nn
import models.norms as norms
import torch
import torch.nn.functional as F

class ApplyNoise(nn.Module):
    def __init__(self, channels,opt):
        super().__init__()
        self.weights = []
        for i in range(opt.class_num):
            self.weights.append(nn.Parameter(torch.zeros(channels,device=torch.device("cuda:"+opt.gpu_ids if torch.cuda.is_available() else "cpu"))))

    def forward(self, x, noise,class_num):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weights[class_num].view(1, -1, 1, 1) * noise.to(x.device)

class AddNoise(nn.Module):
    def __init__(self, channels, opt):
        super().__init__()
        self.noise = ApplyNoise(channels, opt)

    def forward(self, x,noise,class_num):
        x = self.noise(x, noise,class_num)
        return x

class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        self.const_inputs = []
        self.class_num = self.opt.class_num
        for i in range(0, self.class_num):
            z = torch.randn(self.opt.batch_size,self.opt.z_dim, dtype=torch.float32,device=torch.device("cuda:"+opt.gpu_ids if torch.cuda.is_available() else "cpu"))
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, 512, 512)
            self.const_inputs.append(nn.Parameter(z))
        if opt.phase == "test":
            torch.save(self.const_inputs,"const_input_"+self.opt.name+".pt")

        self.addNoise_body = nn.ModuleList([])
        self.noise_inputs = []
        for i in range(0, self.class_num):
            self.noise_inputs.append([])

        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            self.addNoise_body.append(AddNoise(self.channels[i+1],opt=opt))

            if i < self.opt.num_res_blocks - 1:
                for j in range(0, self.class_num):
                    self.noise_inputs[j].append(torch.randn([1, 1 ,2*self.init_W * 2**i, 2*self.init_H * 2**i],device=torch.device("cuda:"+opt.gpu_ids if torch.cuda.is_available() else "cpu")))
            else:
                for j in range(0, self.class_num):
                    self.noise_inputs[j].append(torch.randn([1, 1 ,2*self.init_W * 2**(i-1), 2*self.init_H * 2**(i-1)],device=torch.device("cuda:"+opt.gpu_ids if torch.cuda.is_available() else "cpu")))

        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w
    def forward(self, input,input_class, z=None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda(int(self.opt.gpu_ids))
        if not self.opt.no_3dnoise:
            input_const = self.const_inputs[input_class]
            seg = torch.cat((input_const, seg), dim=1)

        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)

        for i in range(self.opt.num_res_blocks):#num_res_blocks = 6
            x = self.body[i](x, seg)
            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
            x = self.addNoise_body[i](x, self.noise_inputs[input_class][i],input_class)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
