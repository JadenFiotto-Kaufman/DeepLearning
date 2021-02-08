from deeplearning.losses.base import Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class DepthEstimationLoss(Loss):

    def __init__(self,
        no_ssim,
        disable_automasking,
        avg_reprojection,
        disparity_smoothness,
        scales,
        v1_multiscale,
        image_resize,
        frame_ids,
        use_stereo,
        predictive_mask,
        **kwargs):
        super().__init__(**kwargs)

        frame_ids = frame_ids.copy()

        height, width = (image_resize[0], image_resize[0]) if len(image_resize) == 1 else image_resize

        assert height % 32 == 0, "'height' must be a multiple of 32"
        assert width % 32 == 0, "'width' must be a multiple of 32"

        self.height = height
        self.width = width

        assert frame_ids[0] == 0, "frame_ids must start with 0"

        if use_stereo:
            frame_ids.append("s")

        self.frame_ids = frame_ids

        self.disable_automasking = disable_automasking
        self.avg_reprojection = avg_reprojection
        self.disparity_smoothness = disparity_smoothness
        self.scales = scales
        self.v1_multiscale = v1_multiscale
        self.predictive_mask = predictive_mask

        self.num_scales = len(self.scales)

        self.ssim = SSIM() if not no_ssim  else None
            

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if not self.ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def forward(self, inputs_outputs, _):
        inputs, outputs = inputs_outputs

        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            if self.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.height, self.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape))
                loss += weighting_loss.mean()

            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(identity_reprojection_loss.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        return total_loss

    @staticmethod
    def args(parser):
        parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)

        parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        parser.add_argument("--image_resize", nargs='+', type=int, required=True)
        parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")

        super(DepthEstimationLoss, DepthEstimationLoss).args(parser)