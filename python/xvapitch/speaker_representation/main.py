import torch
import torchaudio
import torch.nn as nn
import numpy as np
import librosa

class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
class ResNetSpeakerEncoder(nn.Module):
    """Implementation of the model H/ASP without batch normalization in speaker embedding. This model was proposed in: https://arxiv.org/abs/2009.14153
    Adapted from: https://github.com/clovaai/voxceleb_trainer
    """

    # pylint: disable=W0102
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=True,
        use_torch_spec=True,
        # audio_config=None,
    ):
        super(ResNetSpeakerEncoder, self).__init__()


        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        # self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = torch.nn.Sequential(
                # PreEmphasis(audio_config["preemphasis"]),
                PreEmphasis(0.97),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    # n_fft=audio_config["fft_size"],
                    n_fft=512,
                    win_length=400,
                    hop_length=160,
                    window_fn=torch.hamming_window,
                    n_mels=64,
                ),
            )
        else:
            self.torch_spec = None

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # pylint: disable=R0201
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x.squeeze_(1)
                # if you torch spec compute it otherwise use the mel spec computed by the AP
                if self.use_torch_spec:
                    x = self.torch_spec(x)

                if self.log_input:
                    x = (x + 1e-6).log()
                x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    @torch.no_grad()
    def inference(self, x, l2_norm=False):
        return self.forward(x, l2_norm)

    @torch.no_grad()
    def compute_embedding(self, x, device=None, num_frames=250, num_eval=10, return_mean=True, l2_norm=True):

        if device is None:
            device = self.conv1.weight.device

        x = self.load_and_prepare(x, device)

        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        # map to the waveform size
        if self.use_torch_spec:
            # num_frames = num_frames * self.audio_config["hop_length"]
            num_frames = num_frames * 160

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings

    # def load_checkpoint(self, config: dict, checkpoint_path: str, eval: bool = False, use_cuda: bool = False, cuda_device: int = 0):
    def load_checkpoint(self, checkpoint_path: str, eval: bool = False, use_cuda: bool = False, cuda_device: int = 0):
        # state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        self.eval()

    def load_and_prepare(self, filepath, device):
        wav = load_audio(filepath)
        input_tensor = torch.from_numpy(wav).to(device)
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor

def rms_volume_norm(x, db_level):
    # wav = self._rms_norm(x, db_level)
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(x) * (r ** 2)) / np.sum(x ** 2))
    return x*a

def load_audio(filepath):
    x, sr = librosa.load(filepath, sr=16000)
    # x = rms_volume_norm(x, audio_config["db_level"])
    x = rms_volume_norm(x, -27.0)
    return x



if __name__ == '__main__':

    cuda_device = 1
    device = torch.device(f'cuda:{cuda_device}')

    model = ResNetSpeakerEncoder()
    model = model.to(device)
    model.load_checkpoint(f'speaker_rep.pt')

    # embedding = model.compute_embedding(load_and_prepare(f'./000A2AEB_1.wav', device))
    embedding = model.compute_embedding(f'./000A2AEB_1.wav')
    embedding = embedding.squeeze().cpu().detach().numpy()

    test_orig = np.load(f'./000A2AEB_1.npy')
    print(f'test_orig, {list(test_orig)}')
    print(f'embedding, {list(embedding)}')
    print(f'sum diff, {np.sum(test_orig) - np.sum(embedding)}')