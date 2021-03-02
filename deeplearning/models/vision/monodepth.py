from deeplearning.models.base import Model

import torch
import torch.nn.functional as F

from ._monodepth.networks import (BackprojectDepth, DepthDecoder, PoseCNN,
                                 PoseDecoder, Project3D, ResnetEncoder)
from ._monodepth.util import disp_to_depth, transformation_from_parameters

class DepthEstimator(Model):
    

    def __init__(self, 
        num_layers, 
        weights_init, 
        pose_model_input, 
        pose_model_type,
        predictive_mask, 
        disable_automasking,
        min_depth,
        max_depth,
        v1_multiscale,
        scales,
        frame_ids,
        use_stereo,
        image_resize,
        **kwargs):
        
        frame_ids = frame_ids.copy()

        super().__init__(**kwargs)

        height, width = (image_resize[0], image_resize[0]) if len(image_resize) == 1 else image_resize

        assert height % 32 == 0, "'height' must be a multiple of 32"
        assert width % 32 == 0, "'width' must be a multiple of 32"

        self.height = height
        self.width = width

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.v1_multiscale = v1_multiscale
        self.disable_automasking = disable_automasking
        self.scales = scales
        self.use_stereo = use_stereo

        assert frame_ids[0] == 0, "frame_ids must start with 0"

        if use_stereo:
            frame_ids.append("s")

        self.frame_ids = frame_ids

        self.num_scales = len(self.scales)
        self.num_input_frames = len(self.frame_ids)
        self.num_pose_frames = 2 if pose_model_input == "pairs" else self.num_input_frames


        self.use_pose_net = not (self.use_stereo and self.frame_ids == [0])


        self.encoder = ResnetEncoder(
            num_layers, weights_init == "pretrained")

        self.depth = DepthDecoder(
            self.encoder.num_ch_enc, self.scales)

        self.pose_model_type = pose_model_type

        if self.use_pose_net:
            if pose_model_type == "separate_resnet":
                self.pose_encoder = ResnetEncoder(
                    num_layers,
                    weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)


                self.pose = PoseDecoder(
                   self.pose_encoder.num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif pose_model_type == "shared":
                self.pose = PoseDecoder(
                    self.encoder.num_ch_enc, self.num_pose_frames)

            elif pose_model_type == "posecnn":
                self.pose = PoseCNN(
                    self.num_input_frames if pose_model_input == "all" else 2)


        
        self.backproject_depth = {}
        self.project_3d = {}
        
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(h, w)

            self.project_3d[scale] = Project3D(h, w)

        self.predictive_mask = predictive_mask

        if predictive_mask:
            assert disable_automasking, "When using predictive_mask, please disable automasking with --disable_automasking"

            self.predictive_mask = DepthDecoder(
                self.encoder.num_ch_enc, self.scales,
                num_output_channels=(len(self.frame_ids) - 1))

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.pose_model_type == "separate_resnet":
                        pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                    elif self.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.pose(pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.frame_ids if i != "s"], 1)

                if self.pose_model_type == "separate_resnet":
                    pose_inputs = [self.pose_encoder(pose_inputs)]

            elif self.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.frame_ids if i != "s"]

            axisangle, translation = self.pose(pose_inputs)

            for i, f_i in enumerate(self.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            if self.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.height, self.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def forward(self, inputs):

        if self.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.frame_ids])
            all_features = self.encoder(all_color_aug)
            all_features = [torch.split(f, len(inputs)) for f in all_features]

            features = {}
            for i, k in enumerate(self.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.depth(features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.encoder(inputs["color_aug", 0, 0])
            outputs = self.depth(features)

        if self.predictive_mask:
            outputs["predictive_mask"] = self.predictive_mask(features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        
        return (inputs, outputs)

    def disparity_to_depth(self, disparity, baseline):

        return baseline / (1./self.max_depth + (1./self.min_depth - 1./self.max_depth) * disparity)
        
    @staticmethod
    def args(parser):
        parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])
        parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")

        parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        parser.add_argument("--image_resize", nargs='+', type=int)

        super(DepthEstimator,DepthEstimator).args(parser)
