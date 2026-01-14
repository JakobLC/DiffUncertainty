"""
    implementation of SWAG
"""

import torch
import numpy as np

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

class SWAG(torch.nn.Module):
    def __init__(
        self,
        diag_only=False,     
        max_num_models=30, 
        var_clamp=1e-30, 
    ):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.diag_only = diag_only
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp
        self.param_info = []

    def _ensure_param_buffers(self, model):
        if not self.param_info:
            self._initialize_buffers(model)
        else:
            self._validate_model_structure(model)

    def prepare(self, model):
        """Public wrapper to ensure buffers match a given model."""
        self._ensure_param_buffers(model)

    def _initialize_buffers(self, model):
        self.param_info = []
        for idx, param in enumerate(model.parameters()):
            mean_name = f"buffer_mean_{idx}"
            sq_mean_name = f"buffer_sq_mean_{idx}"
            self.register_buffer(mean_name, torch.zeros_like(param))
            self.register_buffer(sq_mean_name, torch.zeros_like(param))
            cov_name = None
            if not self.diag_only:
                cov_name = f"buffer_cov_mat_sqrt_{idx}"
                self.register_buffer(
                    cov_name, param.new_empty((0, param.numel())).zero_()
                )
            self.param_info.append(
                {
                    "shape": tuple(param.shape),
                    "numel": param.numel(),
                    "mean": mean_name,
                    "sq_mean": sq_mean_name,
                    "cov": cov_name,
                }
            )

    def _validate_model_structure(self, model):
        model_params = list(model.parameters())
        if len(model_params) != len(self.param_info):
            raise ValueError(
                "Model structure does not match the SWAG statistics buffers."
            )
        for param, info in zip(model_params, self.param_info):
            if tuple(param.shape) != info["shape"]:
                raise ValueError(
                    "Model parameter shapes changed and are incompatible with SWAG."
                )

    def _get_buffer(self, info, key):
        name = info.get(key)
        if name is None:
            return None
        return getattr(self, name)

    def sample(
        self,
        model,
        scale=1.0,
        use_low_rank=False,
        blockwise=False,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        if use_low_rank and self.diag_only:
            raise ValueError("Low-rank covariance sampling requested but diag_only=True")

        self._ensure_param_buffers(model)

        if blockwise:
            self.sample_blockwise(model, scale, use_low_rank)
        else:
            self.sample_fullrank(model, scale, use_low_rank)

    def sample_blockwise(self, model, scale, use_low_rank):
        for param, info in zip(model.parameters(), self.param_info):
            mean = self._get_buffer(info, "mean")
            sq_mean = self._get_buffer(info, "sq_mean")
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if use_low_rank:
                cov_mat_sqrt = self._get_buffer(info, "cov")
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                w = mean + scaled_diag_sample + cov_sample
            else:
                w = mean + scaled_diag_sample

            param.data.copy_(w.to(param.device))

    def sample_fullrank(self, model, scale, use_low_rank):
        scale_sqrt = scale ** 0.5
        shared_cov_noise = None
        shared_rank = None

        for param, info in zip(model.parameters(), self.param_info):
            mean = self._get_buffer(info, "mean")
            sq_mean = self._get_buffer(info, "sq_mean")

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
            rand_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

            if use_low_rank:
                cov_mat_sqrt = self._get_buffer(info, "cov")
                if cov_mat_sqrt is not None and cov_mat_sqrt.size(0) > 0:
                    rank = cov_mat_sqrt.size(0)
                    if shared_cov_noise is None or shared_rank != rank:
                        # Reuse a single Gaussian vector so cross-parameter correlations are preserved.
                        shared_cov_noise = cov_mat_sqrt.new_empty(
                            (rank,), requires_grad=False
                        ).normal_()
                        shared_rank = rank
                    cov_sample = cov_mat_sqrt.t().matmul(shared_cov_noise)
                    normalizer = max(self.max_num_models - 1, 1) ** 0.5
                    cov_sample = cov_sample.view_as(mean) / normalizer
                    rand_sample = rand_sample + cov_sample

            sample = mean + scale_sqrt * rand_sample
            param.data.copy_(sample.to(param.device))

    def collect_model(self, base_model):
        self._ensure_param_buffers(base_model)
        n_models = self.n_models.item()
        inv = 1.0 / (n_models + 1.0)
        coeff = n_models * inv

        for info, base_param in zip(self.param_info, base_model.parameters()):
            mean = self._get_buffer(info, "mean")
            sq_mean = self._get_buffer(info, "sq_mean")
            base_data = base_param.data.to(mean.device)

            # first moment
            mean.mul_(coeff)
            mean.add_(base_data * inv)

            # second moment
            sq_mean.mul_(coeff)
            sq_mean.add_(base_data ** 2 * inv)

            # square root of covariance matrix
            if self.diag_only is False:
                cov_mat_sqrt = self._get_buffer(info, "cov")

                # block covariance matrices, store deviation from current mean
                dev = (base_data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                setattr(self, info["cov"], cov_mat_sqrt)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=True):
        if not self.diag_only:
            for info in self.param_info:
                cov_name = info["cov"]
                if cov_name is None or cov_name not in state_dict:
                    continue
                target_tensor = state_dict[cov_name]
                current_tensor = getattr(self, cov_name)
                if current_tensor.shape != target_tensor.shape:
                    setattr(self, cov_name, current_tensor.new_empty(target_tensor.shape))
        super(SWAG, self).load_state_dict(state_dict, strict)

    def export_numpy_params(self, export_cov_mat=False):
        if not self.param_info:
            raise ValueError("SWAG statistics are not initialized.")
        mean_list = []
        sq_mean_list = []
        cov_mat_list = []

        for info in self.param_info:
            mean_tensor = self._get_buffer(info, "mean")
            sq_mean_tensor = self._get_buffer(info, "sq_mean")
            mean_list.append(mean_tensor.cpu().numpy().ravel())
            sq_mean_list.append(sq_mean_tensor.cpu().numpy().ravel())
            if export_cov_mat:
                cov_tensor = self._get_buffer(info, "cov")
                cov_mat_list.append(cov_tensor.cpu().numpy().ravel())
        mean = np.concatenate(mean_list)
        sq_mean = np.concatenate(sq_mean_list)
        var = sq_mean - np.square(mean)

        if export_cov_mat:
            return mean, var, cov_mat_list
        else:
            return mean, var

    def import_numpy_weights(self, model, w):
        self._ensure_param_buffers(model)
        k = 0
        for param, info in zip(model.parameters(), self.param_info):
            mean = self._get_buffer(info, "mean")
            s = info["numel"]
            new_tensor = mean.new_tensor(w[k : k + s].reshape(info["shape"]))
            param.data.copy_(new_tensor)
            k += s

"""

import gpytorch
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

    def generate_mean_var_covar(self):
        if not self.param_info:
            raise ValueError("SWAG statistics are not initialized.")
        mean_list = []
        var_list = []
        cov_mat_root_list = []
        for info in self.param_info:
            mean = self._get_buffer(info, "mean")
            sq_mean = self._get_buffer(info, "sq_mean")
            cov_mat_sqrt = self._get_buffer(info, "cov")
            if cov_mat_sqrt is None:
                cov_mat_sqrt = mean.new_empty((0, info["numel"]))

            mean_list.append(mean)
            var_list.append(sq_mean - mean ** 2.0)
            cov_mat_root_list.append(cov_mat_sqrt)
        return mean_list, var_list, cov_mat_root_list

    def compute_ll_for_block(self, vec, mean, var, cov_mat_root):
        vec = flatten(vec)
        mean = flatten(mean)
        var = flatten(var)

        cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        var_lt = DiagLazyTensor(var + 1e-6)
        covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)
        qdist = MultivariateNormal(mean, covar_lt)

        with gpytorch.settings.num_trace_samples(
            1
        ) and gpytorch.settings.max_cg_iterations(25):
            return qdist.log_prob(vec)

    def block_logdet(self, var, cov_mat_root):
        var = flatten(var)

        cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        var_lt = DiagLazyTensor(var + 1e-6)
        covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)

        return covar_lt.log_det()

    def block_logll(self, param_list, mean_list, var_list, cov_mat_root_list):
        full_logprob = 0
        for i, (param, mean, var, cov_mat_root) in enumerate(
            zip(param_list, mean_list, var_list, cov_mat_root_list)
        ):
            # print('Block: ', i)
            block_ll = self.compute_ll_for_block(param, mean, var, cov_mat_root)
            full_logprob += block_ll

        return full_logprob

    def full_logll(self, param_list, mean_list, var_list, cov_mat_root_list):
        cov_mat_root = torch.cat(cov_mat_root_list, dim=1)
        mean_vector = flatten(mean_list)
        var_vector = flatten(var_list)
        param_vector = flatten(param_list)
        return self.compute_ll_for_block(
            param_vector, mean_vector, var_vector, cov_mat_root
        )

    def compute_logdet(self, block=False):
        _, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if block:
            full_logdet = 0
            for (var, cov_mat_root) in zip(var_list, covar_mat_root_list):
                block_logdet = self.block_logdet(var, cov_mat_root)
                full_logdet += block_logdet
        else:
            var_vector = flatten(var_list)
            cov_mat_root = torch.cat(covar_mat_root_list, dim=1)
            full_logdet = self.block_logdet(var_vector, cov_mat_root)

        return full_logdet

    def diag_logll(self, param_list, mean_list, var_list):
        logprob = 0.0
        for param, mean, scale in zip(param_list, mean_list, var_list):
            logprob += Normal(mean, scale).log_prob(param).sum()
        return logprob

    def compute_logprob(self, model=None, vec=None, block=False, diag=False):
        if not self.param_info:
            if model is None:
                raise ValueError(
                    "Model instance must be provided to initialize SWAG statistics."
                )
            self._ensure_param_buffers(model)

        mean_list, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if vec is None:
            if model is None:
                raise ValueError("Model instance must be provided when vec is None")
            self._ensure_param_buffers(model)
            param_list = [p for p in model.parameters()]
        else:
            param_list = unflatten_like(vec, mean_list)

        if diag:
            return self.diag_logll(param_list, mean_list, var_list)
        elif block is True:
            return self.block_logll(
                param_list, mean_list, var_list, covar_mat_root_list
            )
        else:
            return self.full_logll(param_list, mean_list, var_list, covar_mat_root_list)"""