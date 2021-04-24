import tempfile
from pathlib import Path
import cv2
import torch
import numpy as np
from models.SRResCGAN import Generator

import cog


class Model(cog.Model):

    def setup(self):
        model_path = 'trained_nets_x4/srrescgan_model.pth'  # trained G model of SRResCGAN
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = Generator(scale=4) # SRResCGAN generator net
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    @cog.input("image", type=Path, help="Input image to be upscaled")
    def predict(self, image):
        img_lr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img_LR = torch.from_numpy(np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img_LR.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.no_grad():
            output_SR = self.model(img_LR)
        output_sr = output_SR.data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output_sr = np.transpose(output_sr[[2, 1, 0], :, :], (1, 2, 0))

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), output_sr)

        del img_LR, img_lr
        del output_SR, output_sr
        torch.cuda.empty_cache()

        return out_path
