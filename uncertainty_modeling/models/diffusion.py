#Continuous Guassian Diffusion implemented by Jakob Loenborg Christensen (JakobLC github) jloch@dtu.dk

import enum
import numpy as np
import torch
#from source.utils.mixed import normal_kl, construct_points, nice_split, get_padding_slices
#from source.utils.metric_and_loss import mse_loss,ce1_loss,ce2_loss,ce2_logits_loss
#from source.utils.argparsing import compare_strs
import tqdm

def mse_loss(pred, gt, loss_mask=None, batch_dim=0):
    """mean squared error loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(gt.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = (torch.ones_like(gt)*(1/torch.numel(gt[0]))).to(pred.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred.device)
    return torch.sum(loss_mask*(pred-gt)**2, dim=non_batch_dims)

def bce_loss(pred, gt, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(gt.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = (torch.ones_like(gt)*(1/torch.numel(gt[0]))).to(pred.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred.device)
    likelihood = torch.prod(1-0.5*torch.abs(pred-gt),axis=1,keepdims=True)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def nice_split(s,split_s=",",remove_empty_str=True):
    assert isinstance(s,str), "expected s to be a string"
    assert isinstance(split_s,str), "expected split_s to be a string"
    if len(s)==0:
        out = []
    else:
        out = s.split(split_s)
    if remove_empty_str:
        out = [item for item in out if len(item)>0]
    return out

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def add_(coefs,x,batch_dim=0,flat=False):
    """broadcast and add coefs to x"""
    if isinstance(coefs,np.ndarray):
        coefs = torch.from_numpy(coefs)
    else:
        assert torch.is_tensor(coefs)
    assert torch.is_tensor(x)
    if flat:
        not_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
        return (coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)+x).mean(not_batch_dims)
    else:
        return coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)+x
    
def mult_(coefs,x,batch_dim=0,flat=False):
    """broacast and multiply coefs with x"""
    if isinstance(coefs,np.ndarray):
        coefs = torch.from_numpy(coefs)
    else:
        assert torch.is_tensor(coefs)
    assert torch.is_tensor(x)
    if flat:
        not_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
        return (coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)*x).mean(not_batch_dims)
    else:
        return coefs_(coefs, x.shape, dtype=x.dtype, device=x.device)*x
    
def coefs_(coefs, shape, dtype=torch.float32, device="cuda", batch_dim=0):
    view_shape = [1 for _ in range(len(shape))]
    view_shape[batch_dim] = -1
    return coefs.view(view_shape).to(device).to(dtype)

def get_named_gamma_schedule(schedule_name,b,logsnr_min=-20.0,logsnr_max=20.0):
    #float64 = lambda x: torch.tensor(float(x),dtype=torch.float64)
    if schedule_name=="linear":
        gamma = lambda t: torch.sigmoid(-torch.log(torch.expm1(1e-4+10*t*t)))
    elif schedule_name=="cosine":
        gamma = lambda t: torch.cos(t*torch.pi/2)**2
    elif schedule_name=="linear_simple":
        gamma = lambda t: 1-t
    elif schedule_name=="parabola":
        gamma = lambda t: 1-2*t**2+t**4 #(1-t**2)**2 expanded
    else:
        raise NotImplementedError(schedule_name)
    
    b = (b if torch.is_tensor(b) else torch.tensor(b)).to(torch.float64)
    gamma_wrap1 = input_scaling_wrap(gamma,b)
    slope,bias = logsnr_wrap(gamma_wrap1,logsnr_min,logsnr_max)
    gamma_wrap2 = lambda t: gamma_wrap1((t if torch.is_tensor(t) else torch.tensor(t)).to(torch.float64))*slope+bias
    return gamma_wrap2

def input_scaling_wrap(gamma,b=1.0):
    input_scaling = (b-1.0).abs().item()>1e-9
    if input_scaling:
        gamma_input_scaled = lambda t: b*b*gamma(t)/((b*b-1)*gamma(t)+1)
    else:
        gamma_input_scaled = gamma
    return gamma_input_scaled

def logsnr_wrap(gamma,logsnr_min=-10,logsnr_max=10,dtype=torch.float64):
    if dtype==torch.float64:
        assert logsnr_max<=36, "numerical issues are reached with logsnr_max>36 for float64"
    assert logsnr_min<logsnr_max, "expected logsnr_min<logsnr_max"
    g1_old = gamma(torch.tensor(1,dtype=dtype))
    g0_old = gamma(torch.tensor(0,dtype=dtype))
    g0_new = 1/(1+torch.exp(-torch.tensor(logsnr_max,dtype=dtype)))
    g1_new = 1/(1+torch.exp(-torch.tensor(logsnr_min,dtype=dtype)))
    slope = (g0_new-g1_new)/(g0_old-g1_old)
    bias = g1_new-g1_old*slope
    return slope,bias

def type_from_maybe_str(s,class_type):
    if isinstance(s,class_type):
        return s
    list_of_attribute_strings = [a for a in dir(class_type) if not a.startswith("__")]
    list_of_attribute_strings_lower = [a.lower() for a in list_of_attribute_strings]
    if s.lower() in list_of_attribute_strings_lower:
        s_maybe_not_lower = list_of_attribute_strings[list_of_attribute_strings_lower.index(s.lower())]
        return class_type[s_maybe_not_lower]
    raise ValueError(f"Unknown type: {s}, must be one of {list_of_attribute_strings}")

class ModelPredType(enum.Enum):
    """Which type of output the model predicts. default x"""
    EPS = enum.auto()
    X = enum.auto() 
    V = enum.auto()
    BOTH = enum.auto()

class LossType(enum.Enum):
    """Which type of loss the model uses. default MSE"""
    MSE = enum.auto()  
    BCE = enum.auto()

class VarType(enum.Enum):
    """Time condition type the model uses. default large"""
    small = enum.auto()
    large = enum.auto()

class SamplerType(enum.Enum):
    """How to sample timesteps for training"""
    uniform = enum.auto()
    low_discrepency = enum.auto()
    uniform_low_d = enum.auto()

class ContinuousGaussianDiffusion():
    def __init__(self, 
                 schedule_name="cosine", 
                 input_scale=0.1,
                 model_pred_type="X", 
                 weights_type="sigmoid_-4", 
                 sampler_type="uniform_low_d",
                 var_type="large",
                 loss_type="MSE",
                 logsnr_min=-10.0,
                 logsnr_max=10.0,
                 decouple_loss_weights=True):
        """class to handle the diffusion process"""
        self.loss_type = type_from_maybe_str(loss_type,LossType)
        self.gamma = get_named_gamma_schedule(schedule_name,b=input_scale,logsnr_min=logsnr_min,logsnr_max=logsnr_max)
        self.model_pred_type = type_from_maybe_str(model_pred_type,ModelPredType)
        self.var_type = type_from_maybe_str(var_type,VarType)
        self.weights_type = weights_type
        self.sampler_type = type_from_maybe_str(sampler_type,SamplerType)
        self.decouple_loss_weights = decouple_loss_weights
        
    def snr(self,t):
        """returns the signal to noise ratio, aka alpha^2/sigma^2"""
        return self.gamma(t)/(1-self.gamma(t))
    
    def alpha(self,t):
        """returns the signal coeffecient"""
        return torch.sqrt(self.gamma(t))
    
    def sigma(self,t):
        """returns the noise coeffecient"""
        return torch.sqrt(1-self.gamma(t))
    
    def logsnr(self,t):
        """returns the log signal-to-noise ratio"""
        return torch.log(self.snr(t))

    def diff_logsnr(self,t):
        """returns the derivative of the log signal-to-noise ratio"""
        t_req_grad = torch.autograd.Variable(t, requires_grad = True)
        with torch.enable_grad():
            t_grad = torch.autograd.grad(self.logsnr(t_req_grad).sum(),t_req_grad,create_graph=True)[0]
        t_grad.detach_()
        return t_grad

    def loss_weights(self, t):
        snr = self.snr(t)
        if self.weights_type=="SNR":
            weights = snr
        elif self.weights_type=="SNR_plus1":
            weights = 1+snr
        elif self.weights_type=="SNR_trunc":
            weights = torch.maximum(snr,torch.ones_like(snr))
        elif self.weights_type=="uniform":
            weights = torch.ones_like(snr)
        elif self.weights_type.startswith("sigmoid"): # aka sigmoid loss from simpler diffusion/VDM++
            if self.weights_type=="sigmoid":
                bias = 0# aka simply the gamma function
            else:
                bias = float(self.weights_type.split("_")[1])
            weights = torch.sigmoid(self.logsnr(t)+bias)
        else:
            raise NotImplementedError(self.weights_type)

        if self.decouple_loss_weights:
            weights *= -self.diff_logsnr(t)
        return weights
    
    def sample_t(self,bs):
        if self.sampler_type==SamplerType.uniform:
            t = torch.rand(bs)
        elif self.sampler_type==SamplerType.low_discrepency:
            t0 = torch.rand()/bs
            t = (torch.arange(bs)/bs+t0)
            t = t[torch.randperm(bs)]
        elif self.sampler_type==SamplerType.uniform_low_d:
            t = (torch.randperm(bs)+torch.rand(bs))/bs
        else:
            raise NotImplementedError(self.sampler_type)
        return t

    def get_losses(self, pred_x, x, t, loss_mask=None):
        """computes the losses given predictions and ground truths"""
        loss_weights = self.loss_weights(t)
        if self.loss_type==LossType.MSE:
            losses = mult_(loss_weights,mse_loss(pred_x,x,loss_mask))
        elif self.loss_type==LossType.BCE:
            losses = mult_(loss_weights,bce_loss(pred_x,x,loss_mask))
        else:
            raise NotImplementedError(self.loss_type)
        return losses
    
    def train_loss_step(self, model, x, im, loss_mask=None, eps=None, t=None, self_cond=False):
        """compute one training step and return the loss
        model must be a diffusion model. x is the ground truth and im is the image to condition on.
        """
        if self_cond:
            raise NotImplementedError("self conditioning during training not implemented yet")

        t = self.sample_t(x.shape[0]).to(x.device)

        eps = torch.randn_like(x)
 
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        x_t = mult_(alpha_t,x) + mult_(sigma_t,eps)
        """if any(self_cond):
            with torch.no_grad():
                output = model(x_t, t)
                pred_x, pred_eps = self.get_predictions(output,x_t,alpha_t,sigma_t)
                model_kwargs['self_cond'] = [(pred_x[i] if self_cond[i] else None) for i in range(len(x))]
        else:
            model_kwargs['self_cond'] = None"""
        x_t_with_image = torch.cat([x_t,im],dim=1)
        output = model(x_t_with_image, t)
        
        pred_x, pred_eps = self.get_predictions(output,x_t,alpha_t,sigma_t)

        losses = self.get_losses(pred_x, x, t, loss_mask)
        loss = torch.mean(losses)

        return loss, pred_x
    
    def get_x_from_eps(self,eps,x_t,alpha_t,sigma_t):
        """returns the predicted x from eps"""
        #return (1/alpha_t)*(x_t-sigma_t*eps)
        return mult_(1/alpha_t,x_t) - mult_(sigma_t/alpha_t,eps)
    
    def get_eps_from_x(self,x,x_t,alpha_t,sigma_t):
        """returns the predicted eps from x"""
        #return (1/sigma_t)*(x_t-alpha_t*x)
        return mult_(1/sigma_t,x_t) - mult_(alpha_t/sigma_t,x)
    
    def get_predictions(self, output, x_t, alpha_t, sigma_t, clip_x=False,
                        guidance_weight=None,model_output_guidance=None):
        """returns predictions based on the equation x_t = alpha_t*x + sigma_t*eps"""
        if self.model_pred_type==ModelPredType.EPS:
            pred_eps = output
            if guidance_weight is None:
                pred_x = self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.X:
            pred_x = output
            pred_eps = self.get_eps_from_x(pred_x,x_t,alpha_t,sigma_t)
        elif self.model_pred_type==ModelPredType.BOTH:
            pred_eps, pred_x = torch.split(output, output.shape[1]//2, dim=1)
            #reconsiles the two predictions (parameterized by eps and by direct prediction):
            pred_x = mult_(alpha_t,pred_x)+mult_(sigma_t,self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t))
        elif self.model_pred_type==ModelPredType.V:
            #V = alpha*eps-sigma*x
            v = output
            pred_x = mult_(alpha_t,x_t) - mult_(sigma_t,v)
            pred_eps = self.get_eps_from_x(pred_x,x_t,alpha_t,sigma_t)
        else:
            raise NotImplementedError(self.model_pred_type)

        if guidance_weight is not None:
            pred_eps = (1+guidance_weight)*pred_eps - guidance_weight*self.get_predictions(model_output_guidance,x_t,
                                                                                           alpha_t,sigma_t,
                                                                                           clip_x=False,
                                                                                           replace_padding=False)[1]
            pred_x = self.get_x_from_eps(pred_eps,x_t,alpha_t,sigma_t)
        if clip_x:
            assert not pred_x.requires_grad
            pred_x = torch.clamp(pred_x,-1,1)
            #pred_eps = (1/sigma_t)*(x_t-alpha_t*pred_x) Should this be done? TODO
        return pred_x, pred_eps
        
    def ddim_step(self, i, pred_x, pred_eps, num_steps):
        logsnr_s = self.logsnr(torch.tensor(i / num_steps))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        x_s_pred = alpha_s * pred_x + sigma_s * pred_eps
        if i==0:
            return pred_x
        else:
            return x_s_pred

    def ddpm_step(self, i, pred_x, x_t, num_steps):
        t = torch.tensor((i + 1.) / num_steps).to(pred_x.dtype)
        s = torch.tensor(i / num_steps).to(pred_x.dtype)
        x_s_dist = self.p_distribution(
            x_t=x_t,
            pred_x=pred_x,
            logsnr_t=self.logsnr(t),
            logsnr_s=self.logsnr(s))
        if i==0:
            return pred_x
        else:
            return x_s_dist['mean'] + x_s_dist['std'] * torch.randn_like(x_t)
        
    def sample_loop(self, model, x_init, im, num_steps, sampler_type="ddpm", clip_x=False,
                    guidance_weight=0.0, progress_bar=False, self_cond=False):
        if self_cond:
            raise NotImplementedError("self conditioning during sampling not implemented yet")
        if sampler_type == 'ddim':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddim_step(i, pred_x, pred_eps, num_steps)
        elif sampler_type == 'ddpm':
            body_fun = lambda i, pred_x, pred_eps, x_t: self.ddpm_step(i, pred_x, x_t, num_steps)
        else:
            raise NotImplementedError(sampler_type)
        
        guidance_weight = transform_guidance_weight(guidance_weight,x_init)
        if progress_bar:
            trange = tqdm.tqdm(range(num_steps-1, -1, -1), desc="Batch progress.")
        else:
            trange = range(num_steps-1, -1, -1)

        x_t = x_init
        
        for i in trange:
            t = torch.tensor((i + 1.) / num_steps)
            alpha_t, sigma_t = self.alpha(t), self.sigma(t)
            t_cond = t.to(x_t.dtype).to(x_t.device)
            
            if guidance_weight is not None:
                #model_output_guidance = model(x_t, t_cond, **{k: v for (k,v) in model_kwargs.items() if k in nice_split(guidance_kwargs)})
                raise NotImplementedError("guidance not implemented yet")
            else:
                model_output_guidance = None
            x_t_with_image = torch.cat([x_t,im],dim=1) # Conditioning image should be added here if needed
            model_output = model(x_t_with_image, t_cond)
            pred_x, pred_eps = self.get_predictions(output=model_output,
                                                    x_t=x_t,
                                                    alpha_t=alpha_t,
                                                    sigma_t=sigma_t,
                                                    clip_x=clip_x,
                                                    guidance_weight=guidance_weight,
                                                    model_output_guidance=model_output_guidance)
            #print 20% and 80% quantiles of pred_x
            #print("Step",i,"pred_x 20%:",torch.quantile(pred_x,0.2).item(),"80%:",torch.quantile(pred_x,0.8).item())
            #if any(self_cond): model_kwargs['self_cond'] = [(pred_x[i] if self_cond[i] else None) for i in range(len(x_t))]
            
            x_t = body_fun(i, pred_x, pred_eps, x_t)
        #assert x_t.shape == x_init.shape and x_t.dtype == x_init.dtype fails with mixed precision
        return x_t
    
    def p_distribution(self, x_t, pred_x, logsnr_t, logsnr_s):
        """computes p(x_s | x_t)."""
        out = self.q_distribution(
            x_t=x_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
            x=pred_x)
        out['pred_bit'] = pred_x
        return out
    
    def q_distribution(self, x, x_t, logsnr_s, logsnr_t, x_logvar=None):
        """computes q(x_s | x_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
        alpha_st = torch.sqrt((1. + torch.exp(-logsnr_t)) / (1. + torch.exp(-logsnr_s)))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
        one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
        log_one_minus_r = torch.log1p(-torch.exp(logsnr_s - logsnr_t))  # log(1-SNR(t)/SNR(s))

        mean = mult_(r * alpha_st, x_t) + mult_(one_minus_r * alpha_s, x)
        if x_logvar is None:
            x_logvar = self.var_type
        if x_logvar==VarType.small:
            # same as setting x_logvar to -infinity
            var = one_minus_r * torch.sigmoid(-logsnr_s)
            logvar = log_one_minus_r + torch.nn.LogSigmoid()(-logsnr_s)
        elif x_logvar==VarType.large:
            # same as setting x_logvar to nn.LogSigmoid()(-logsnr_t)
            var = one_minus_r * torch.sigmoid(-logsnr_t)
            logvar = log_one_minus_r + torch.nn.LogSigmoid()(-logsnr_t)
        else:        
            raise NotImplementedError(self.var_type)
        return {'mean': mean, 'std': torch.sqrt(var), 'var': var, 'logvar': logvar}

def transform_guidance_weight(gw, x):
    if gw is None:
        return None
    else:
        bs = x.shape[0]
        device = x.device
        dtype = x.dtype
        w = torch.tensor(gw, dtype=dtype, device=device) if not torch.is_tensor(gw) else gw
        if w.numel() != bs:
            assert w.numel() == 1, f"guidance_weight must be a scalar or batch_size={bs} got {str(w.numel())}"
            if abs(w)<1e-9:
                return None
            w = w.repeat(bs)
        else:
            if (abs(w)<1e-9).all():
                return None
        assert w.numel() == bs, f"guidance_weight must be a scalar or batch_size={bs} got {str(w.numel())}"
        w = w.view(bs,1,1,1)
        return w
