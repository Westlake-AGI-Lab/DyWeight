import torch
from torch_utils import persistence
import solvers
from solver_utils import get_schedule
from training.inception import compute_inception_mse_loss
from training.inception import InceptionFeatureExtractor
import torch.nn.functional as F

#----------------------------------------------------------------------------

def get_solver_fn(solver_name):
    if solver_name == 'dyweight':
        solver_fn = solvers.dyweight_sampler
    elif solver_name == 'heun':
        solver_fn = solvers.heun_sampler
    elif solver_name == 'dpm':
        solver_fn = solvers.dpm_sampler
    elif solver_name == 'dpmpp':
        solver_fn = solvers.dpmpp_sampler
    elif solver_name == 'euler':
        solver_fn = solvers.euler_sampler
    elif solver_name == 'ipndm':
        solver_fn = solvers.ipndm_sampler
    elif solver_name == 'unipc':
        solver_fn = solvers.unipc_sampler
    else:
        raise ValueError("Got wrong solver name {}".format(solver_name))
    return solver_fn

#----------------------------------------------------------------------------

@persistence.persistent_class
class DyWeight_loss:
    def __init__(
        self, 
        num_steps=None, 
        sampler_stu=None, 
        sampler_tea=None, 
        teacher_steps=None,
        schedule_type=None, 
        schedule_rho=None, 
        afs=False, 
        max_order=None, 
        sigma_min=None, 
        sigma_max=None, 
        predict_x0=True, 
        lower_order_final=True,
        loss_type='l2',
        huber_delta=0.1,
        dataset_name=None,
    ):
        self.num_steps = num_steps
        self.solver_stu = get_solver_fn(sampler_stu)
        self.solver_tea = get_solver_fn(sampler_tea)
        self.teacher_steps = teacher_steps
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.max_order = max_order
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.dataset_name = dataset_name
        
        self.t_steps = None
        self.inception_extractor = None
        if self.loss_type == 'inception':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.inception_extractor = InceptionFeatureExtractor(device=device)
    
    def _ensure_inception_extractor(self, device):
        """Recreate inception_extractor if it was removed during checkpoint saving."""
        if self.loss_type == 'inception' and self.inception_extractor is None:
            self.inception_extractor = InceptionFeatureExtractor(device=device)

    def get_final_teacher_output(self, net, tensor_in, labels=None, condition=None, unconditional_condition=None):
        if self.t_steps is None:
            self.t_steps = get_schedule(self.num_steps, self.sigma_min, self.sigma_max, 
                                     schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, 
                                     device=tensor_in.device, net=net)
        
        final_output = self.solver_tea(
            net, 
            tensor_in / self.t_steps[0], 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition, 
            num_steps=self.teacher_steps, 
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=False, 
            denoise_to_zero=True,
            return_inters=False,
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
        )
        return final_output
    
    def get_student_final_output(self, net, tensor_in, labels=None, hybrid_weights=None, t_scale=None, 
                               condition=None, unconditional_condition=None, use_afs=False):
        if self.t_steps is None:
            self.t_steps = get_schedule(self.num_steps, self.sigma_min, self.sigma_max, 
                                     schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, 
                                     device=tensor_in.device, net=net)
        
        final_output = self.solver_stu(
            net, 
            tensor_in / self.t_steps[0], 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition, 
            num_steps=self.num_steps, 
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=use_afs, 
            return_inters=False,
            hybrid_weights=hybrid_weights,
            t_scale=t_scale,
        )
        return final_output
    
    def compute_loss(self, student_final, teacher_final, net=None, **kwargs):
        if self.loss_type == 'l2':
            return F.mse_loss(student_final, teacher_final, reduction='none').mean()
        elif self.loss_type == 'l1':
            return F.l1_loss(student_final, teacher_final, reduction='none').mean()
        elif self.loss_type == 'huber':
            return F.huber_loss(student_final, teacher_final, reduction='none', delta=self.huber_delta).mean()
        elif self.loss_type == 'inception':
            # Ensure inception_extractor is initialized (in case it was removed during checkpoint saving)
            self._ensure_inception_extractor(student_final.device)
            
            if self.dataset_name in ['lsun_bedroom_ldm', 'ms_coco']:
                if net is None:
                    raise ValueError("Network model is required for LDM dataset inception loss")
                with torch.no_grad():
                    teacher_decoded = net.model.decode_first_stage(teacher_final)
                student_decoded = net.model.differentiable_decode_first_stage(student_final)
                teacher_norm = ((teacher_decoded + 1) * 127.5).clamp(0, 255)
                student_norm = ((student_decoded + 1) * 127.5).clamp(0, 255)
            else:
                teacher_norm = ((teacher_final + 1) * 127.5).clamp(0, 255)
                student_norm = ((student_final + 1) * 127.5).clamp(0, 255)
            return compute_inception_mse_loss(student_norm, teacher_norm, self.inception_extractor).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
