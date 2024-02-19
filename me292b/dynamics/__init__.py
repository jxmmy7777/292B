from typing import Union

from me292b.dynamics.single_integrator import SingleIntegrator
from me292b.dynamics.unicycle import Unicycle
from me292b.dynamics.bicycle import Bicycle
from me292b.dynamics.double_integrator import DoubleIntegrator
from me292b.dynamics.base import Dynamics, DynType


def get_dynamics_model(dyn_type: Union[str, DynType]):
    if dyn_type in ["Unicycle", DynType.UNICYCLE]:
        return Unicycle
    elif dyn_type == ["SingleIntegrator", DynType.SI]:
        return SingleIntegrator
    elif dyn_type == ["DoubleIntegrator", DynType.DI]:
        return DoubleIntegrator
    else:
        raise NotImplementedError("Dynamics model {} is not implemented".format(dyn_type))
