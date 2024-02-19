
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
        feat = self.map_model(map_inputs)
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