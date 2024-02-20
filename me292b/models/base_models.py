
import numpy as np
import math
import textwrap
from collections import OrderedDict
from typing import Dict, Union, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign

import me292b.utils.tensor_utils as TensorUtils
import me292b.dynamics as dynamics
from me292b.utils.batch_utils import batch_utils
from me292b.utils.tensor_utils import reshape_dimensions, flatten
from me292b.utils.loss_utils import trajectory_loss, goal_reaching_loss, collision_loss
class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            feature_dim: int = None,
            output_activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = input_image_shape[0]
        self._feature_dim = feature_dim
        if output_activation is None:
            self._output_activation = nn.Identity()
        else:
            self._output_activation = output_activation()

        # configure conv backbone
        if model_arch == "resnet18":
            self.map_model = resnet18()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (512, out_h, out_w)

            pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_out_dim = self.conv_out_shape[0]
            self.map_model.conv1 = nn.Conv2d(
                self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
            self.map_model.avgpool = pooling
            
            if feature_dim is not None:
                self.map_model.fc = nn.Linear(
                    in_features=self.pool_out_dim, out_features=feature_dim)
            else:
                self.map_model.fc = nn.Identity()
        elif model_arch == "mobilenet_v2":
            mobilenet = mobilenet_v2()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            # self.conv_out_shape = (512, out_h, out_w)
            mobilenet.features[0][0] = nn.Conv2d(
                self.num_input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            # self.map_model = mobilenet.features

            # The output features of MobileNetV2 is 1280 for the last layer
            self.conv_out_shape = (1280, out_h, out_w)
            pooling = nn.AdaptiveAvgPool2d((1, 1))
            # Replace the default average pooling layer with an adaptive one
            mobilenet.avgpool = pooling
            self.pool_out_dim = self.conv_out_shape[0]
            
            # If you are not using the classifier part of mobilenet, you can discard it
            # Otherwise, you will need to adjust the classifier to match your `feature_dim`
            mobilenet.classifier = nn.Sequential(
                nn.Linear(self.conv_out_shape[0], feature_dim),
                output_activation()
            )
            
            # Assign the modified mobilenet back to self.map_model
            self.map_model = mobilenet
           

    def output_shape(self, input_shape=None):
        if self._feature_dim is not None:
            return [self._feature_dim]
        else:
            return [self.pool_out_dim]

    def feature_channels(self):
        if self.model_arch in ["resnet18", "resnet34"]:
            channels = OrderedDict({
                "layer1": 64,
                "layer2": 128,
                "layer3": 256,
                "layer4": 512,
            })
        else:
            channels = OrderedDict({
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
            })
        return channels

    def feature_scales(self):
        return OrderedDict({
            "layer1": 1/4,
            "layer2": 1/8,
            "layer3": 1/16,
            "layer4": 1/32
        })

    def forward(self, map_inputs) -> torch.Tensor:
        feat = self.map_model(map_inputs) # [B, self._feature_dim]
        feat = self._output_activation(feat) 
        return feat


class TrajectoryDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            state_dim: int = 3,
            num_steps: int = None,
            dynamics_type: Union[str, dynamics.DynType] = None,
            dynamics_kwargs: dict = None,
            step_time: float = None,
            network_kwargs: dict = None,
            Gaussian_var = False
    ):
        """
        A class that predict future trajectories based on input features
        Args:
            feature_dim (int): dimension of the input feature
            state_dim (int): dimension of the output trajectory at each step
            num_steps (int): (optional) number of future state to predict
            dynamics_type (str, dynamics.DynType): (optional) if specified, the network predicts action
                for the dynamics model instead of future states. The actions are then used to predict
                the future trajectories.
            step_time (float): time between steps. required for using dynamics models
            network_kwargs (dict): keyword args for the decoder networks
            Gaussian_var (bool): whether output the variance of the predicted trajectory
        """
        super(TrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.step_time = step_time
        self._network_kwargs = network_kwargs
        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self.Gaussian_var = Gaussian_var
        self._create_dynamics()
        self._create_networks()

    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        elif self._dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=self._dynamics_kwargs["acce_bound"],
                ddh_bound=self._dynamics_kwargs["ddh_bound"],
                max_hdot=self._dynamics_kwargs["max_yawvel"],
                max_speed=self._dynamics_kwargs["max_speed"]
            )
        else:
            self.dyn = None

    def _create_networks(self):
        raise NotImplementedError

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        raise NotImplementedError

    def _forward_dynamics(self, current_states, actions,**kwargs):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        bound = True if "predict" in kwargs and kwargs["predict"] else False
        x, pos, yaw = self.dyn.forward_dynamics(
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time,
            bound = bound,
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj,x

    def forward(self, inputs, current_states=None, num_steps=None,**kwargs):
        preds = self._forward_networks(
            inputs, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            preds["controls"] = preds["trajectories"]
            preds["trajectories"], x = self._forward_dynamics(
                current_states=current_states,
                actions=preds["trajectories"],
                **kwargs
            )
            preds["terminal_state"] = x[...,-1,:]
        return preds
    

class MLPTrajectoryDecoder(TrajectoryDecoder):
    def _create_networks(self):
        net_kwargs = dict() if self._network_kwargs is None else dict(self._network_kwargs)
        if self._network_kwargs is None:
            net_kwargs = dict()
        
        assert isinstance(self.num_steps, int)
        if self.dyn is None:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.state_dim))
        else:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.dyn.udim))
        if self.Gaussian_var:
            pred_shapes["logvar"] = (self.num_steps, self.state_dim)

        state_as_input = net_kwargs.pop("state_as_input")
        if self.dyn is not None:
            assert state_as_input   # TODO: deprecated, set default to True and remove from configs

        if state_as_input and self.dyn is not None:
            feature_dim = self.feature_dim + self.dyn.xdim
        else:
            feature_dim = self.feature_dim

        self.mlp = SplitMLP(
            input_dim=feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        if self._network_kwargs["state_as_input"] and self.dyn is not None:
            inputs = torch.cat((inputs, current_states), dim=-1)

        if inputs.ndim == 2:
            # [B, D]
            preds = self.mlp(inputs)
        elif inputs.ndim == 3:
            # [B, A, D]
            preds = TensorUtils.time_distributed(inputs, self.mlp)
        else:
            raise ValueError(
                "Expecting inputs to have ndim == 2 or 3, got {}".format(inputs.ndim))
        return preds
    
    
class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
    ) -> None:

        super().__init__()
        self.map_encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=map_feature_dim,
            output_activation=nn.ReLU
        )
        self.traj_decoder = trajectory_decoder
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        dec_output = self.traj_decoder.forward(inputs=map_feat, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]
            out_dict["curr_states"] = curr_states
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        # pred_edges = batch_utils().get_edges_from_batch(
        #     data_batch=data_batch,
        #     ego_predictions=pred_batch["predictions"]
        # )
        #
        # coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(
            prediction_loss=pred_loss,
            # goal_loss=goal_loss,
            # collision_loss=coll_loss
        )
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses
    
    
    
    
    
    
    
    
    
    
    # ---------------base models ----------------
    

class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_dim (int): dimension of outputs
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        act = None if self._act is None else self._act.__name__
        output_act = None if self._output_act is None else self._output_act.__name__

        indent = ' ' * 4
        msg = "input_dim={}\noutput_shape={}\nlayer_dims={}\nlayer_func={}\ndropout={}\nact={}\noutput_act={}".format(
            self._input_dim, self.output_shape(), self._layer_dims,
            self._layer_func.__name__, self._dropouts, act, output_act
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg


class SplitMLP(MLP):
    """
    A multi-output MLP network: The model split and reshapes the output layer to the desired output shapes
    """

    def __init__(
            self,
            input_dim: int,
            output_shapes: OrderedDict,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_shapes (dict): named dictionary of output shapes
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """

        assert isinstance(output_shapes, OrderedDict)
        output_dim = 0
        for v in output_shapes.values():
            output_dim += np.prod(v)
        self._output_shapes = output_shapes

        super(SplitMLP, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            layer_dims=layer_dims,
            layer_func=layer_func,
            layer_func_kwargs=layer_func_kwargs,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation
        )

    def output_shape(self, input_shape=None):
        return self._output_shapes

    def forward(self, inputs):
        outs = super(SplitMLP, self).forward(inputs)
        out_dict = dict()
        ind = 0
        for k, v in self._output_shapes.items():
            v_dim = int(np.prod(v))
            out_dict[k] = reshape_dimensions(
                outs[:, ind: ind + v_dim], begin_axis=1, end_axis=2, target_dims=v)
            ind += v_dim
        return out_dict