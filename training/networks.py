import torch
from torch_utils import persistence
import torch.nn as nn


@persistence.persistent_class
class DyWeight_predictor(torch.nn.Module):
    """
    DyWeight predictor for hybrid derivative weights and time scaling.
    """
    def __init__(
        self,
        num_steps               = 10,
        max_history_steps       = None,
        init_mode               = 'ipndm',
        enable_t_scale_learning = True,
        t_scale_init_mode       = 'ones',
        afs                     = False,
        guidance_rate           = None,
        guidance_type           = None,
        **kwargs
    ):
        super().__init__()
        
        assert num_steps >= 2, "num_steps must be at least 2"
        assert init_mode in ['ipndm', 'uniform', 'perturbed', 'euler'], f"Unknown init_mode: {init_mode}"
        assert t_scale_init_mode in ['ones', 'uniform', 'perturbed'], f"Unknown t_scale_init_mode: {t_scale_init_mode}"
        
        self.num_steps = num_steps
        self.max_history_steps = max_history_steps
        self.init_mode = init_mode
        self.enable_t_scale_learning = enable_t_scale_learning
        self.t_scale_init_mode = t_scale_init_mode
        self.afs = afs
        self.guidance_rate = guidance_rate
        self.guidance_type = guidance_type
        
        if max_history_steps is None:
            self.effective_window_size = num_steps - 1
        else:
            self.effective_window_size = max_history_steps - 1
        
        self.hybrid_weights = nn.Parameter(torch.zeros(num_steps - 1, num_steps - 1))
        
        self.register_buffer('mask', torch.zeros(num_steps - 1, num_steps - 1))
        for i in range(num_steps - 1):
            start_idx = max(0, i - self.effective_window_size)
            end_idx = i + 1
            self.mask[i, start_idx:end_idx] = 1.0
        
        if enable_t_scale_learning:
            t_scale_dim = num_steps - 2 if afs else num_steps - 1
            self.t_scale = nn.Parameter(torch.ones(t_scale_dim))
        else:
            self.t_scale = None
        
        self._initialize_parameters()

    def _initialize_parameters(self):
        with torch.no_grad():
            if self.init_mode == 'ipndm':
                for i in range(self.num_steps - 1):
                    available_steps = min(i + 1, self.effective_window_size + 1)
                    if available_steps == 1:
                        self.hybrid_weights[i, i] = 1.0
                    elif available_steps == 2:
                        self.hybrid_weights[i, i] = 3.0/2.0
                        self.hybrid_weights[i, i-1] = -1.0/2.0
                    elif available_steps == 3:
                        self.hybrid_weights[i, i] = 23.0/12.0
                        self.hybrid_weights[i, i-1] = -16.0/12.0
                        self.hybrid_weights[i, i-2] = 5.0/12.0
                    else:
                        coeffs = [55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0]
                        for j in range(min(4, available_steps)):
                            self.hybrid_weights[i, i-j] = coeffs[j]
            
            elif self.init_mode == 'uniform':
                for i in range(self.num_steps - 1):
                    available_steps = min(i + 1, self.effective_window_size + 1)
                    start_idx = max(0, i - self.effective_window_size)
                    uniform_value = 1.0 / available_steps
                    for j in range(start_idx, i + 1):
                        self.hybrid_weights[i, j] = uniform_value
                        
            elif self.init_mode == 'perturbed':
                for i in range(self.num_steps - 1):
                    available_steps = min(i + 1, self.effective_window_size + 1)
                    if available_steps == 1:
                        self.hybrid_weights[i, i] = 1.0
                    elif available_steps == 2:
                        self.hybrid_weights[i, i] = 3.0/2.0
                        self.hybrid_weights[i, i-1] = -1.0/2.0
                    elif available_steps == 3:
                        self.hybrid_weights[i, i] = 23.0/12.0
                        self.hybrid_weights[i, i-1] = -16.0/12.0
                        self.hybrid_weights[i, i-2] = 5.0/12.0
                    else:
                        coeffs = [55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0]
                        for j in range(min(4, available_steps)):
                            self.hybrid_weights[i, i-j] = coeffs[j]
                perturbation = (torch.rand_like(self.hybrid_weights) - 0.5) * 0.1
                self.hybrid_weights.data += perturbation
                
            elif self.init_mode == 'euler':
                for i in range(self.num_steps - 1):
                    self.hybrid_weights[i, i] = 1.0
            
            self.hybrid_weights.data *= self.mask
            
            if self.enable_t_scale_learning and self.t_scale is not None:
                if self.t_scale_init_mode == 'ones':
                    self.t_scale.data.fill_(1.0)
                elif self.t_scale_init_mode == 'uniform':
                    self.t_scale.data.uniform_(0.8, 1.2)
                elif self.t_scale_init_mode == 'perturbed':
                    perturbation = (torch.rand_like(self.t_scale) - 0.5) * 0.1
                    self.t_scale.data = 1.0 + perturbation

    def forward(self, *args, **kwargs):
        masked_weights = self.hybrid_weights * self.mask
        if self.enable_t_scale_learning and self.t_scale is not None:
            return masked_weights, self.t_scale
        else:
            return masked_weights, None
    
    def get_effective_window_size(self):
        return self.effective_window_size
    
    def get_step_weights(self, step_idx):
        masked_weights = self.hybrid_weights * self.mask
        return masked_weights[step_idx]
    
    def get_t_scale_value(self, scale_idx):
        if self.enable_t_scale_learning and self.t_scale is not None:
            return self.t_scale[scale_idx]
        else:
            return 1.0
            
    def extra_repr(self):
        return (f'num_steps={self.num_steps}, '
                f'max_history_steps={self.max_history_steps}, '
                f'init_mode={self.init_mode}, '
                f'enable_t_scale_learning={self.enable_t_scale_learning}, '
                f't_scale_init_mode={self.t_scale_init_mode}')
