
from .latte.modeling_latte import Latte_models
from .latte.modeling_latte_i2v import Latte_i2v_models

Diffusion_models = {}
Diffusion_models.update(Latte_models)
Diffusion_models.update(Latte_i2v_models)

    