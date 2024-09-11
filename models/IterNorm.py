"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from py_scripts.redact import redact


class IterNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, X: torch.Tensor, running_mean: torch.Tensor,
                running_wmat: torch.Tensor, nc: int, T: int, eps: float, momentum: float, training: bool, **kwargs):
        ctx.T = T
        # change N x C x H x W to (G x D) x (N x H x W), i.e. g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P: list[torch.Tensor] = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)

        return Xn

    # This is cursed unreadable code territory
    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        _, _, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(1.5, -0.5, g_tmp, P2)
            g_P.baddbmm_(1, -0.5, P2, g_tmp)
            g_P.baddbmm_(1, -0.5, P[k - 1].matmul(g_tmp), P[k - 1])
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()

        return grad_input, None, None, None, None, None, None, None


class IterNormRotation(torch.nn.Module):
    """
    Concept Whitening Module

    The Whitening part is adapted from IterNorm. The core of the CW module is learning an extra rotation matrix R that
    aligns target concepts with the output feature maps.

    Because the concept activation is calculated based on a feature map, which is a matrix,
    there are multiple ways to calculate the activation, denoted by activation_mode.
    """
    def __init__(self, num_features, concept_mat: torch.Tensor, latent_mappings, num_channels=None, T=10, dim=4,
                 eps=1e-5, momentum=0.05, cw_lambda=0.1, affine=False, mode=-1, activation_mode='pool_max'):
        super(IterNormRotation, self).__init__()
        assert dim == 4, 'IterNormRotation does not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.concept_mat = concept_mat
        self.latent_mappings = latent_mappings
        self.cw_lambda = cw_lambda
        self.affine = affine
        self.dim = dim
        self.mode = mode
        self.activation_mode = activation_mode

        # num_features and num_channels are the dimensionality of the latent space.
        if num_channels is None:
            num_channels = num_features

        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        self.weight = Parameter(torch.Tensor(*shape))
        self.bias = Parameter(torch.Tensor(*shape))

        # pooling and unpooling used in gradient computation
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.maxunpool = torch.nn.MaxUnpool2d(kernel_size=3, stride=3)

        # intermediate values for hook extraction
        self.current_batch_X_hat = None
        self.current_batch_X_rot_activated = None

        # loss counter, number of samples for each concept
        self.concept_counter: dict[int, int] = {}
        self.low_concept_loss: dict[int, torch.Tensor] = {}
        self.high_concept_loss: dict[int, torch.Tensor] = {}

        # running mean
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_channels).expand(1, num_channels, num_channels))
        # running rotation matrix
        self.register_buffer('running_rot', torch.eye(num_channels))
        # sum Gradient, need to take average later
        self.register_buffer('sum_G', torch.zeros(num_channels, num_channels))
        # grad counter, number of gradient for each concept
        self.register_buffer("grad_counter", torch.ones(num_channels) * 0.001)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def reset_counters(self):
        self.grad_counter = (torch.ones(self.num_channels) * 0.001).cuda()
        self.concept_counter = {}
        self.low_concept_loss = {}
        self.high_concept_loss = {}

    def update_rotation_matrix(self):
        """
        Update the rotation matrix R using the accumulated gradient G.
        The update uses Cayley transform to make sure R is always orthonormal.
        """
        with torch.no_grad():
            G: torch.Tensor = self.sum_G / self.grad_counter.reshape(-1, 1)
            R: torch.Tensor = self.running_rot.clone()

            for _ in range(2):
                # constants
                tau = 1000  # learning rate in Cayley transform
                alpha = 0
                beta = 100000000
                c1 = 1e-4
                c2 = 0.9

                A = torch.einsum('in,jn->ij', G, R) - torch.einsum('in,jn->ij', R, G)  # GR^T - RG^T
                Id = torch.eye(self.num_channels).cuda()
                dF_0 = -0.5 * (A ** 2).sum()

                # binary search for appropriate learning rate
                iteration = 0
                while True:
                    Q = torch.mm((Id + 0.5 * tau * A).inverse(), Id - 0.5 * tau * A)
                    Y_tau = torch.mm(Q, R)
                    F_X = (G * R).sum()
                    F_Y_tau = (G * Y_tau).sum()
                    dF_tau = -torch.mm(torch.einsum('ni,nj->ij', G, (Id + 0.5 * tau * A).inverse()),
                                       torch.mm(A, 0.5 * (R + Y_tau))).trace()

                    if F_Y_tau > F_X + c1 * tau * dF_0 + 1e-18:
                        beta = tau
                        tau = (beta + alpha) / 2
                    elif dF_tau + 1e-18 < c2 * dF_0:
                        alpha = tau
                        tau = (beta + alpha) / 2
                    else:
                        break

                    iteration += 1
                    if iteration > 500:
                        print(
                            "--------------------update fail------------------------\n"
                            f"F_Y_tau: {F_Y_tau}, F_X + c1 * tau * dF_0: {F_X + c1 * tau * dF_0}\n"
                            f"dF_tau: {dF_tau}, c2 * dF_0: {c2 * dF_0}\n"
                            "-------------------------------------------------------"
                        )
                        break

                Q = torch.mm((Id + 0.5 * tau * A).inverse(), Id - 0.5 * tau * A)
                R = torch.mm(Q, R)

            self.running_rot = R
            self.reset_counters()

    def forward(self, X, X_redact_coords=None, orig_x_dim=None):
        # Applying concept whitening
        X_hat = IterNorm.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T,
                               self.eps, self.momentum, self.training)

        # Saving the intermediate value for hooks during evaluation
        if not self.training:
            self.current_batch_X_hat = X_hat

        # Updating the gradient matrix, using the concept dataset
        # The gradient is accumulated with momentum to stabilize the training
        with torch.no_grad():
            # When mode >= 0, the mode-th column of gradient matrix is accumulated
            # Throughout this code, d = dimensionality of latent space, self.mode = index of current concept
            if self.mode >= 0:
                X_redacted = redact(X_hat, X_redact_coords, orig_x_dim)
                X_test = torch.einsum('bchw,dc->bdhw', X_redacted, self.running_rot)

                # Applying the concept activation function
                if self.activation_mode == 'mean':
                    # bd
                    redact_bool = (X_redacted != 0).to(X_redacted)
                    X_activated = X_redacted.sum((2, 3)) / (redact_bool.sum((2, 3)) + 0.0001)

                elif self.activation_mode == 'max':
                    # bdhw
                    max_values = torch.max(torch.max(X_test, 2, keepdim=True)[0], 3, keepdim=True)[0]
                    max_bool = (max_values == X_test).to(X_redacted)
                    # bd
                    X_activated = (X_redacted * max_bool).sum((2, 3)) / max_bool.sum((2, 3))

                elif self.activation_mode == 'pos_mean':
                    # bdhw
                    pos_bool = (X_test > 0).to(X_redacted)
                    # bd
                    X_activated = (X_redacted * pos_bool).sum((2, 3)) / (pos_bool.sum((2, 3)) + 0.0001)

                elif self.activation_mode == 'pool_max':
                    # bdhw
                    maxpool_value, maxpool_indices = self.maxpool(X_test)
                    # bdhw
                    X_test_unpool: torch.Tensor = self.maxunpool(maxpool_value, maxpool_indices,
                                                                 output_size=X_test.size())
                    maxpool_bool = ((X_test == X_test_unpool) & (X_test != 0)).to(X_redacted)
                    # bd
                    X_activated = (X_redacted * maxpool_bool).sum((2, 3)) / (maxpool_bool.sum((2, 3)) + 0.0001)

                # Calculating the projections onto higher level concept subspaces
                num_concepts = self.concept_mat.size()[0]
                # TODO: integrate latent concept mappings into this
                concept_mask = (
                    F.pad(self.concept_mat[self.mode], (0, self.num_channels - num_concepts), mode='constant', value=0)
                ).unsqueeze(0).cuda()

                X_rot = torch.einsum('bc,dc->bd', X_activated, self.running_rot)
                # bd
                X_rot_masked = X_rot * concept_mask
                # NOTE: takes the max values, or 0 if all values are 0 or negative
                # X_rot_masked[X_rot_masked == 0] = float('-inf')
                max_values = torch.max(X_rot_masked, 1, keepdim=True)[0]
                max_bool = (max_values == X_rot_masked).to(X_activated)
                collapsed_max_values = torch.max(X_rot_masked, 1)[0]

                # Calculating the gradient matrix of the concept activation loss
                # For grad and self.sum_G, each ROW corresponds to a concept
                low_grad = -X_activated.mean((0,))
                # dc, high_grad
                grad_mode = self.latent_mappings[self.mode]
                grad = -(torch.einsum('bd,bm->bmd', X_activated, max_bool)).mean((0,)) * self.cw_lambda / \
                    self.concept_mat[self.mode].sum()
                grad[grad_mode, :] += low_grad + grad[grad_mode, :]

                # Updating the gradient matrix G
                self.sum_G = self.momentum * grad + (1. - self.momentum) * self.sum_G
                self.grad_counter[grad_mode] += 1

                # Updating the CW loss
                if self.mode not in self.concept_counter:
                    self.concept_counter[self.mode] = torch.as_tensor(0.0001)
                    self.low_concept_loss[self.mode] = torch.as_tensor(0.0)
                    self.high_concept_loss[self.mode] = torch.as_tensor(0.0)

                self.concept_counter[self.mode] += X_test.size(0)

                low_concept_loss = X_activated[:, grad_mode].sum()
                self.low_concept_loss[self.mode] = (self.low_concept_loss[self.mode].cpu() + low_concept_loss.cpu()).cuda()

                high_concept_loss = collapsed_max_values.sum()
                self.high_concept_loss[self.mode] = (self.high_concept_loss[self.mode].cpu() + high_concept_loss.cpu()).cuda()

        # We set mode = -1 when we don't need to update G. For example, when we train for main objective
        X_hat = torch.einsum('bchw,dc->bdhw', X_hat, self.running_rot)
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return 'num_features={num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)
