import torch
from spikingjelly.clock_driven import neuron
from typing import Callable
from spikingjelly.clock_driven import surrogate
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
try:
    import cupy
    from spikingjelly.clock_driven import cu_kernel_opt
    from spikingjelly import configure
    from spikingjelly.clock_driven import neuron_kernel
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    cupy = None
    cu_kernel_opt = None
    configure = None
    neuron_kernel = None

tab4_str = '\t\t\t\t'  # used for aligning code
curly_bracket_l = '{'
curly_bracket_r = '}'


class sigmoid_mlif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if torch.cuda.is_available():
            out = torch.zeros_like(x[:, ::2, ...]).cuda()
        else:
            out = torch.zeros_like(x[:, ::2, ...])
        cond1 = x[:, ::2, ...]
        cond2 = x[:, 1::2, ...]
        out[torch.logical_and(cond1 > 0, cond2 >= 0)] = 1.0
        if x.requires_grad:
            L = torch.tensor([alpha])
            ctx.save_for_backward(x, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x, out, others) = ctx.saved_tensors
        alpha = others[0].item()
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x)
            if x.shape[1] > 2:
                for i in range(x.shape[1] // 2 - 1):
                    sgax = (x[:, 2 * i, ...] * alpha).sigmoid_()
                    grad_x[:, 2 * i, ...] = grad_output[:, i, ...] * (1. - sgax) * sgax * alpha
                    sgax = (x[:, 2 * i + 1, ...] * alpha).sigmoid()
                    grad_x[:, 2 * i + 1, ...] = - grad_output[:, i, ...] * (1. - sgax) * sgax * alpha
                sgax = (x[:, 2 * (i + 1), ...] * alpha).sigmoid_()
                grad_x[:, 2 * (i + 1), ...] = grad_output[:, i + 1, ...] * (1. - sgax) * sgax * alpha
            else:
                sgax = (x[:, 0, ...] * alpha).sigmoid_()
                grad_x[:, 0, ...] = grad_output[:, 0, ...] * (1. - sgax) * sgax * alpha
            
        return grad_x, None

class SigmoidMLIF(surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):

        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid_mlif.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid_mlif()

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        return ""

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)

class MLIFNode(neuron.BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = SigmoidMLIF(),
                 detach_reset: bool = False, channels: int = 1, scaled=False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input
        self.channels = channels
        self.scaled = scaled

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - 1. / self.tau) + x
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x

    def neuronal_fire(self):
        rst = torch.zeros_like(self.v)
        temp_mem = []

        if self.channels == 1:
            temp_mem.append(self.v / self.v_threshold - 1.0)
            temp_mem.append(self.v / self.v_threshold - 1.0)
            rst += (temp_mem[0] > 0).float()
        else:
            if self.scaled:
                for out_ch in range(self.channels - 1):
                    temp_mem.append(self.v / ((2 ** (out_ch - self.channels + 1)) * self.v_threshold) - 1.0)
                    temp_mem.append(1.0 - self.v / ((2 ** (out_ch + 1 - self.channels + 1)) * self.v_threshold))
                    rst = torch.logical_or(rst, (torch.logical_and(temp_mem[2 * out_ch] > 0, temp_mem[2 * out_ch + 1] >= 0))).float()
                temp_mem.append(self.v / ((2 ** (out_ch + 1 - self.channels + 1)) * self.v_threshold) - 1.0)
                temp_mem.append(self.v / ((2 ** (out_ch + 1 - self.channels + 1)) * self.v_threshold) - 1.0)
                rst = torch.logical_or(rst, (temp_mem[2 * (out_ch + 1)] > 0)).float()
            else:
                for out_ch in range(self.channels - 1):
                    temp_mem.append(self.v / ((2 ** out_ch) * self.v_threshold) - 1.0)
                    temp_mem.append(1.0 - self.v / (self.v_threshold * (2 ** (out_ch + 1))))
                    rst = torch.logical_or(rst, (torch.logical_and(temp_mem[2 * out_ch] > 0, temp_mem[2 * out_ch + 1] >= 0))).float()
                temp_mem.append(self.v / ((2 ** (out_ch + 1)) * self.v_threshold) - 1.0)
                temp_mem.append(self.v / ((2 ** (out_ch + 1)) * self.v_threshold) - 1.0)
                rst = torch.logical_or(rst, (temp_mem[2 * (out_ch + 1)] > 0)).float()

        return self.surrogate_function(torch.stack(temp_mem, dim=1)), rst

    def neuronal_reset(self, rst):
        if self.detach_reset:
            rst_d = rst.detach()
        else:
            rst_d = rst

        if self.v_reset is None:
            # soft reset
            self.v = self.v - rst_d * self.v_threshold
        else:
            # hard reset
            self.v = (1. - rst_d) * self.v + rst_d * self.v_reset

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        temp_spike, rst = self.neuronal_fire()
        self.neuronal_reset(rst)
        spike = torch.zeros_like(self.v)
        if self.scaled:
            for out_ch in range(self.channels):
                spike += (2 ** (out_ch - self.channels + 1)) * temp_spike[:, out_ch, ...]
        else:
            for out_ch in range(self.channels):
                spike += (2 ** out_ch) * temp_spike[:, out_ch, ...]
        
        return spike

class MultiStepMLIFNode(MLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = SigmoidMLIF(),
                 detach_reset: bool = False, backend='torch', channels: int = 1, scaled=False):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, channels, scaled)
        self.register_memory('v_seq', None)

        neuron.check_backend(backend)

        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq
        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def reset(self):
        super().reset()

class FastMultiStepMLIFNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(decay_input: bool, hard_reset: bool, dtype: str, channels, scaled, kernel_name_prefix: str = 'FastMLIFNode'):
        kernel_name = f'{kernel_name_prefix}_fptt_decayInput{decay_input}_{"hard" if hard_reset else "soft"}Reset_{dtype}'

        if dtype == 'fp32':
            code = rf'''
            extern "C" __global__
            void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
            const float & reciprocal_tau, 
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {
                    const int dt = neuron_num;
                    for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                    {
                        const int t = index + mem_offset;
            '''

            if hard_reset:
                if decay_input:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                    '''
                else:
                    code += r'''
                        h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                    '''

                if scaled:
                    if channels == 1:
                        code += r'''
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                        '''
                    else:
                        left = str(float(2 ** (- channels + 1)))
                        right = str(float(2 ** (1 - channels + 1)))
                        code += r'''
                        if ((h_seq[t] >= (v_threshold * %s)) && (h_seq[t] < (v_threshold * %s)))
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = v_reset;
                            //v_v_seq[t + dt] = (1.0f - spike_seq[t]) * h_seq[t] + spike_seq[t] * v_reset;
                        }
                        ''' % (left, right, left)
                        for i in range(1, channels - 1):
                            left = str(float(2 ** (i - channels + 1)))
                            right = str(float(2 ** ((i + 1) - channels + 1)))
                            code += r'''
                        else if ((h_seq[t] >= (%s * v_threshold)) && (h_seq[t] < (v_threshold * %s))) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = v_reset;
                            //v_v_seq[t + dt] = (1.0f - spike_seq[t]) * h_seq[t] + spike_seq[t] * v_reset;
                        }
                            ''' % (left, right, left)
                        i = channels - 1
                        left = str(float(2 ** (i - channels + 1)))
                        code += r'''
                        else if (h_seq[t] >= (%s * v_threshold)) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = v_reset;
                            //v_v_seq[t + dt] = (1.0f - spike_seq[t]) * h_seq[t] + spike_seq[t] * v_reset;
                        }
                        ''' % (left, left)
                else:
                    if channels == 1:
                        code += r'''
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = v_reset;
                        }
                        '''
                    else:
                        left = str(float(1))
                        right = str(float(2))
                        code += r'''
                        if ((h_seq[t] >= (v_threshold * %s)) && (h_seq[t] < (v_threshold * %s)))
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = v_reset;
                            //v_v_seq[t + dt] = (1.0f - spike_seq[t]) * h_seq[t] + spike_seq[t] * v_reset;
                        }
                        ''' % (left, right, left)
                        for i in range(1, channels - 1):
                            left = str(float(2 ** i))
                            right = str(float(2 ** (i + 1)))
                            code += r'''
                        else if ((h_seq[t] >= (%s * v_threshold)) && (h_seq[t] < (v_threshold * %s))) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = v_reset;
                            //v_v_seq[t + dt] = (1.0f - spike_seq[t]) * h_seq[t] + spike_seq[t] * v_reset;
                        }
                            ''' % (left, right, left)
                        i = channels - 1
                        left = str(float(2 ** i))
                        code += r'''
                        else if (h_seq[t] >= (%s * v_threshold)) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = v_reset;
                            //v_v_seq[t + dt] = (1.0f - spike_seq[t]) * h_seq[t] + spike_seq[t] * v_reset;
                        }
                        ''' % (left, left)

            else:
                if decay_input:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                    '''

                if scaled:
                    if channels == 1:
                        code += r'''
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                        '''
                    else:
                        left = str(float(2 ** (- channels + 1)))
                        right = str(float(2 ** (1 - channels + 1)))
                        code += r'''
                        if ((h_seq[t] >= (v_threshold * %s)) && (h_seq[t] < (v_threshold * %s)))
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold * %sf;
                        }
                        ''' % (left, right, left, left)
                        for i in range(1, channels - 1):
                            left = str(float(2 ** (i - channels + 1)))
                            right = str(float(2 ** ((i + 1) - channels + 1)))
                            code += r'''
                        else if ((h_seq[t] >= (%s * v_threshold)) && (h_seq[t] < (v_threshold * %s))) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold * %sf;
                        }
                            ''' % (left, right, left, left)
                        i = channels - 1
                        left = str(float(2 ** (i - channels + 1)))
                        code += r'''
                        else if (h_seq[t] >= (%s * v_threshold)) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold * %sf;
                        }
                        ''' % (left, left, left)
                else:
                    if channels == 1:
                        code += r'''
                        if (h_seq[t] >= v_threshold)
                        {
                            spike_seq[t] = 1.0f;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold;
                        }
                        '''
                    else:
                        left = str(float(1))
                        right = str(float(2))
                        code += r'''
                        if ((h_seq[t] >= (v_threshold * %s)) && (h_seq[t] < (v_threshold * %s)))
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold * %sf;
                        }
                        ''' % (left, right, left, left)
                        for i in range(1, channels - 1):
                            left = str(float(2 ** i))
                            right = str(float(2 ** (i + 1)))
                            code += r'''
                        else if ((h_seq[t] >= (%s * v_threshold)) && (h_seq[t] < (v_threshold * %s))) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold * %sf;
                        }
                            ''' % (left, right, left, left)
                        i = channels - 1
                        left = str(float(2 ** i))
                        code += r'''
                        else if (h_seq[t] >= (%s * v_threshold)) 
                        {
                            spike_seq[t] = %sf;
                            v_v_seq[t + dt] = h_seq[t] - v_threshold * %sf;
                        }
                        ''' % (left, left, left)

            code += r'''
                        else
                        {
                            spike_seq[t] = 0.0f;
                            v_v_seq[t + dt] = h_seq[t];
                        }
    
                    }
                }
            }
            '''
        elif dtype == 'fp16':
            code = rf'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
            const half & reciprocal_tau,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel) 
            '''

            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {
                const int numel_2 = numel >> 1;
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                {
                    const int t = index + mem_offset;
            '''
            if hard_reset:
                if decay_input:
                    code += r'''
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                        h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                    '''

                if scaled:
                    if channels == 1:
                        code += r'''
                            spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                            v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                        '''
                    else:
                        code += r'''
                            spike_seq[t] = '''
                        str_list = []
                        for i in range(channels - 1):
                            left = str(float(2 ** (i - channels + 1)))
                            right = str(float(2 ** ((i + 1) - channels + 1)))
                            str_list.append(r'''__hmul2(__hsub2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2))), __float2half2_rn(%sf))''' % (left, right, left))
                        i = channels - 1
                        left = str(float(2 ** (i - channels + 1)))
                        str_list.append(r'''__hmul2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __float2half2_rn(%sf))''' % (left, left))

                        for i in range(channels - 1):
                            code += r'''__hadd2(''' + str_list[i] + ''',
                                           '''
                        code += r'''        ''' + str_list[channels - 1]
                        for i in range(channels - 1):
                            code += r''')'''
                        code += ''';
                                '''
                        
                        # code += r'''
                        #     v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                        # '''

                        code += r'''
                            v_v_seq[t + stride] = __hadd2(__hmul2(__hgeu2(h_seq[t], __hmul2(v_threshold_half2, __float2half2_rn(%sf))), v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), __hgeu2(h_seq[t], __hmul2(v_threshold_half2, __float2half2_rn(%sf)))), h_seq[t]));
                        ''' % (str(float(2 ** (- channels + 1))), str(float(2 ** (-channels + 1))))
                else:
                    if channels == 1:
                        code += r'''
                            spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                            v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                        '''
                    else:
                        code += r'''
                            spike_seq[t] = '''
                        str_list = []
                        for i in range(channels - 1):
                            left = str(float(2 ** i))
                            right = str(float(2 ** (i + 1)))
                            str_list.append(r'''__hmul2(__hsub2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2))), __float2half2_rn(%sf))''' % (left, right, left))
                        i = channels - 1
                        left = str(float(2 ** i))
                        str_list.append(r'''__hmul2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __float2half2_rn(%sf))''' % (left, left))

                        for i in range(channels - 1):
                            code += r'''__hadd2(''' + str_list[i] + ''',
                                           '''
                        code += r'''        ''' + str_list[channels - 1]
                        for i in range(channels - 1):
                            code += r''')'''
                        code += ''';
                                '''

                        # code += r'''
                        #     v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                        # '''

                        code += r'''
                           v_v_seq[t + stride] = __hadd2(__hmul2(__hgeu2(h_seq[t], __hmul2(v_threshold_half2, __float2half2_rn(%sf))), v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), __hgeu2(h_seq[t], __hmul2(v_threshold_half2, __float2half2_rn(%sf)))), h_seq[t]));
                        ''' % (str(float(2 ** 0)), str(float(2 ** 0)))

            else:
                if decay_input:
                    code += r'''
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                    '''

                if scaled:
                    if channels == 1:
                        code += r'''
                            spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                            v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                        '''
                    else:
                        code += r'''
                            spike_seq[t] = '''
                        str_list = []
                        for i in range(channels - 1):
                            left = str(float(2 ** (i - channels + 1)))
                            right = str(float(2 ** ((i + 1) - channels + 1)))
                            str_list.append(r'''__hmul2(__hsub2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2))), __float2half2_rn(%sf))''' % (left, right, left))
                        i = channels - 1
                        left = str(float(2 ** (i - channels + 1)))
                        str_list.append(r'''__hmul2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __float2half2_rn(%sf))''' % (left, left))

                        for i in range(channels - 1):
                            code += r'''__hadd2(''' + str_list[i] + ''',
                                           '''
                        code += r'''        ''' + str_list[channels - 1]
                        for i in range(channels - 1):
                            code += r''')'''
                        code += ''';
                                '''

                        code += r'''
                            v_v_seq[t + stride] = __hsub2(h_seq[t], __hmul2(v_threshold_half2, spike_seq[t]));
                        '''

                else:
                    if channels == 1:
                        code += r'''
                            spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                            v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                        '''
                    else:
                        code += r'''
                            spike_seq[t] = '''
                        str_list = []
                        for i in range(channels - 1):
                            left = str(float(2 ** i))
                            right = str(float(2 ** (i + 1)))
                            str_list.append(r'''__hmul2(__hsub2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2))), __float2half2_rn(%sf))''' % (left, right, left))
                        i = channels - 1
                        left = str(float(2 ** i))
                        str_list.append(r'''__hmul2(__hgeu2(h_seq[t], __hmul2(__float2half2_rn(%sf), v_threshold_half2)), __float2half2_rn(%sf))''' % (left, left))

                        for i in range(channels - 1):
                            code += r'''__hadd2(''' + str_list[i] + ''',
                                           '''
                        code += r'''        ''' + str_list[channels - 1]
                        for i in range(channels - 1):
                            code += r''')'''
                        code += ''';
                                '''

                        code += r'''
                            v_v_seq[t + stride] = __hsub2(h_seq[t], __hmul2(v_threshold_half2, spike_seq[t]));
                        '''
            code += r'''
                }
            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, decay_input: bool, hard_reset: bool, detach_reset: bool, dtype: str):

        kernel_name = f'FastMLIFNode_bptt_decayInput{decay_input}_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_last,
            const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''

            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        //const float over_th = h_seq[t] - v_threshold;
                        const float over_th = h_seq[t];
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                        //const float grad_v_to_h = 1.0f - spike_seq[t];
                        float grad_v_to_h;
                        if (spike_seq[t] > 0.0f) {
                            grad_v_to_h = 0.0f;
                        } else {
                            grad_v_to_h = 1.0f;
                        }
                        '''
                else:
                    code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f;
                        '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                        //const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                        float grad_v_to_h;
                        if (spike_seq[t] > 0.0f) {
                            grad_v_to_h = (v_reset - h_seq[t]) * grad_s_to_h;
                        } else {
                            grad_v_to_h = 1.0f - (v_reset - h_seq[t]) * grad_s_to_h;
                        }
                        // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                        '''
                else:
                    code_grad_v_to_h = r'''
                        const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                        // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                        '''

            code += code_grad_v_to_h
            code += r'''
                        grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                        // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
                        '''
            if decay_input:
                code += r'''
                        grad_x_seq[t] = grad_h * reciprocal_tau;
                        '''
            else:
                code += r'''
                        grad_x_seq[t] = grad_h;
                        '''
            code += r'''
                    }
                    grad_v_last[index] = grad_h * one_sub_reciprocal_tau;
                }
            }
            '''
        elif dtype == 'fp16':
            code = fr'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_last,
            const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {   
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                {
                    const int t = index + mem_offset;

                    //const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
                    const half2 over_th = h_seq[t];
            '''

            code += code_grad_s_to_h

            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    //const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hgtu2(spike_seq[t], __float2half2_rn(0.0f)));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    //const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), __hgtu2(spike_seq[t], __float2half2_rn(0.0f))));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    '''

            code += code_grad_v_to_h
            code += r'''                        
                    grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
            '''
            if decay_input:
                code += r''' 
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                '''
            else:
                code += r''' 
                        grad_x_seq[t] = grad_h;
                '''
            code += r'''
                }
            grad_v_last[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_last: torch.Tensor, decay_input: bool, tau: float, v_threshold: float,
                v_reset: float, detach_reset: bool, sg_cuda_code_fun, channels, scaled):
        requires_grad = x_seq.requires_grad or v_last.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        elif x_seq.dtype == torch.float16:
            dtype = 'fp16'
            cp_dtype = np.half
        else:
            raise NotImplementedError

        use_pad = False
        if dtype == 'fp16' and v_last.numel() % 2 != 0:
            # only fp16 needs even numel because we use half2 to accelerate
            # when numel is odd, we will pad x_seq
            use_pad = True
            x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
            v_last = F.pad(v_last, (0, 1))  # [N] -> [N + 1]

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_last.unsqueeze(0), v_seq))

        with cu_kernel_opt.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads
            if dtype == 'fp16':
                assert neuron_num % 2 == 0
                blocks = cu_kernel_opt.cal_blocks(neuron_num >> 1)
                # we will take two neurons to calculate as one neuron in cuda half2
            else:
                blocks = cu_kernel_opt.cal_blocks(neuron_num)

            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            cp_reciprocal_tau = cupy.asarray(1. / tau, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1. / tau, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num,
                               cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cu_kernel_opt.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num,
                    cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset,
                               cp_neuron_num, cp_numel]

            kernel = FastMultiStepMLIFNodePTT.create_fptt_kernel(decay_input, hard_reset, dtype, channels, scaled)
            kernel(
                (blocks,), (threads,),
                cu_kernel_opt.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.decay_input = decay_input
            ctx.use_pad = use_pad
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_reciprocal_tau = cp_reciprocal_tau
            ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        if ctx.use_pad:
            # grad_spike_seq.shape = [T, N]
            # grad_v_seq.shape = [T, N]
            # h_seq.shape = [T, N + 1]
            # spike_seq.shape = [T, N + 1]
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()
        if configure.save_spike_as_bool_in_neuron_kernel:
            h_seq = ctx.saved_tensors[0]
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        else:
            h_seq, spike_seq = ctx.saved_tensors
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_last = zero_data[-1]

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        elif grad_spike_seq.dtype == torch.float16:
            dtype = 'fp16'
        else:
            raise NotImplementedError

        kernel = FastMultiStepMLIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, ctx.decay_input, hard_reset,
                                                        ctx.detach_reset, dtype)

        with cu_kernel_opt.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau,
                    ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num,
                    ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last,
                               ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                               ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cu_kernel_opt.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last, ctx.cp_reciprocal_tau,
                    ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_last,
                               ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                               ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cu_kernel_opt.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        if ctx.use_pad:
            return grad_x_seq[..., :-1], grad_v_last[..., :-1], None, None, None, None, None, None, None, None
        else:
            return grad_x_seq, grad_v_last, None, None, None, None, None, None, None, None

class fast_sigmoid_mlif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, mlif_channels, scaled):

        spike = x.clone()

        if scaled:
            spike[x > 1] = 1
            idx = x < 2 ** (1 - mlif_channels)
            spike = 2 ** torch.floor(torch.log2(spike))
            spike[idx] = 0
        else:
            spike[x >= (2 ** (mlif_channels - 1))] = (2 ** (mlif_channels - 1))
            idx = x < 1
            spike = 2 ** torch.floor(torch.log2(spike))
            spike[idx] = 0

        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.mlif_channels = mlif_channels
            ctx.scaled = scaled

        return spike

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        x = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(grad_output)

            if ctx.scaled:
                for i in range(ctx.mlif_channels):
                    sgax = ((x - 2 ** (i - ctx.mlif_channels + 1)) * ctx.alpha).sigmoid_()
                    if i == 0:
                        grad_x += (2 ** (- ctx.mlif_channels + 1)) * (1. - sgax) * sgax * ctx.alpha
                    else:
                        grad_x += (2 ** ((i - 1) - ctx.mlif_channels + 1)) * (1. - sgax) * sgax * ctx.alpha
            else:
                for i in range(ctx.mlif_channels):
                    sgax = ((x - 2 ** i) * ctx.alpha).sigmoid_()  # * math.ceil(2 ** (i-1))
                    grad_x += math.ceil(2 ** (i-1)) * (1. - sgax) * sgax * ctx.alpha

            grad_x = grad_output * grad_x

        return grad_x, None, None, None

class FastSigmoidMLIF(surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True, mlif_channels=1, scaled=False):
        super().__init__(alpha, spiking)
        self.mlif_channels = mlif_channels
        self.scaled = scaled

    @staticmethod
    def spiking_function(x, alpha, mlif_channels, scaled):
        return fast_sigmoid_mlif.apply(x, alpha, mlif_channels, scaled)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid_mlif()

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
        {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            if self.scaled:
                for i in range(self.mlif_channels):
                    mlif_str = "mlif" + str(i)
                    shift_str = str(float(2 ** (i - self.mlif_channels + 1)))
                    code += f'''
        {tab4_str}const float {mlif_str}_{sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * ({x} - {shift_str}f)));
        '''
                code += f'''
        {tab4_str}const float {y} =
        '''
                for i in range(self.mlif_channels - 1):
                    mlif_str = "mlif" + str(i)
                    if i == 0:
                        scale_str = str(float((2 ** (- self.mlif_channels + 1))))
                    else:
                        scale_str = str(float((2 ** ((i - 1) - self.mlif_channels + 1))))
                    code += f'''
                            {scale_str}f * (1.0f - {mlif_str}_{sg_name}_sigmoid_ax) * {mlif_str}_{sg_name}_sigmoid_ax * {alpha} + 
                            '''
                i = self.mlif_channels - 1
                mlif_str = "mlif" + str(i)
                scale_str = str(float((2 ** ((i - 1) - self.mlif_channels + 1))))
                code += f'''
                            {scale_str}f * (1.0f - {mlif_str}_{sg_name}_sigmoid_ax) * {mlif_str}_{sg_name}_sigmoid_ax * {alpha};
                        '''
            else:
                for i in range(self.mlif_channels):
                    mlif_str = "mlif" + str(i)
                    shift_str = str(2 ** i)
                    code += f'''
        {tab4_str}const float {mlif_str}_{sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * ({x} - {shift_str}.0f)));
        '''
                code += f'''
        {tab4_str}const float {y} =
        '''
                for i in range(self.mlif_channels - 1):
                    mlif_str = "mlif" + str(i)
                    scale_str = str(math.ceil(2 ** (i - 1)))
                    code += f'''
                            {scale_str}.0f * (1.0f - {mlif_str}_{sg_name}_sigmoid_ax) * {mlif_str}_{sg_name}_sigmoid_ax * {alpha} + 
                            '''
                i = self.mlif_channels - 1
                mlif_str = "mlif" + str(i)
                scale_str = str(math.ceil(2 ** (i - 1)))
                code += f'''
                            {scale_str}.0f * (1.0f - {mlif_str}_{sg_name}_sigmoid_ax) * {mlif_str}_{sg_name}_sigmoid_ax * {alpha};
                        '''
        elif dtype == 'fp16':
            if self.scaled:
                code += f'''
        {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
        '''
                for i in range(self.mlif_channels):
                    mlif_str = "mlif" + str(i)
                    shift_str = str(float(2 ** (i - self.mlif_channels + 1)))
                    code += f'''
        {tab4_str}const half2 {mlif_str}_{sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, __hsub2({x}, __float2half2_rn({shift_str}f))))), __float2half2_rn(1.0f)));
        '''

                code += f'''
        {tab4_str}const half2 {y} = '''
                str_list = []
                for i in range(self.mlif_channels):
                    if i == 0:
                        scale_str = str(float((2 ** (- self.mlif_channels + 1))))
                    else:
                        scale_str = str(float((2 ** ((i - 1) - self.mlif_channels + 1))))
                    mlif_str = "mlif" + str(i)
                    str_list.append(f'''__hmul2(__hmul2(__hmul2(__float2half2_rn({scale_str}f), __hsub2(__float2half2_rn(1.0f), {mlif_str}_{sg_name}_sigmoid_ax)), {mlif_str}_{sg_name}_sigmoid_ax), {sg_name}_alpha)''')

                for i in range(self.mlif_channels - 1):
                    code += r'''__hadd2(''' + str_list[i] + ''',
                                        '''
                code += r'''        ''' + str_list[self.mlif_channels - 1]
                for i in range(self.mlif_channels - 1):
                    code += r''')'''
                code += ''';
                        '''
            else:
                code += f'''
        {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
        '''
                for i in range(self.mlif_channels):
                    mlif_str = "mlif" + str(i)
                    shift_str = str(2 ** i)
                    code += f'''
        {tab4_str}const half2 {mlif_str}_{sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, __hsub2({x}, __float2half2_rn({shift_str}.0f))))), __float2half2_rn(1.0f)));
        '''

                code += f'''
        {tab4_str}const half2 {y} = '''
                str_list = []
                for i in range(self.mlif_channels):
                    mlif_str = "mlif" + str(i)
                    scale_str = str(math.ceil(2 ** (i - 1)))
                    str_list.append(f'''__hmul2(__hmul2(__hmul2(__float2half2_rn({scale_str}.0f), __hsub2(__float2half2_rn(1.0f), {mlif_str}_{sg_name}_sigmoid_ax)), {mlif_str}_{sg_name}_sigmoid_ax), {sg_name}_alpha)''')

                for i in range(self.mlif_channels - 1):
                    code += r'''__hadd2(''' + str_list[i] + ''',
                                        '''
                code += r'''        ''' + str_list[self.mlif_channels - 1]
                for i in range(self.mlif_channels - 1):
                    code += r''')'''
                code += ''';
                        '''
        else:
            raise NotImplementedError
        code += f'''
        {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha, self.mlif_channels, self.scaled)
        else:
            return self.primitive_function(x, self.alpha)

class FastMLIFNode(neuron.BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = FastSigmoidMLIF(),
                 detach_reset: bool = False, channels: int = 1, scaled=False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input
        self.channels = channels
        self.surrogate_function = FastSigmoidMLIF(mlif_channels=channels, scaled=scaled)
        self.scaled = scaled

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - 1. / self.tau) + x
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v / self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v[spike_d > 0.] = self.v_reset

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        return spike

class FastMultiStepMLIFNode(FastMLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = FastSigmoidMLIF(),
                 detach_reset: bool = False, backend='torch', channels: int = 1, scaled=False):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, channels, scaled)
        self.register_memory('v_seq', None)

        neuron.check_backend(backend)

        self.backend = backend
        self.channels = channels
        self.scaled = scaled

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = FastMultiStepMLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold,
                self.v_reset, self.detach_reset, self.surrogate_function.cuda_code, self.channels, self.scaled
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def reset(self):
        super().reset()
