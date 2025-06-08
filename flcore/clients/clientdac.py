import torch
import torch.nn as nn
import copy
import time
import numpy as np
from flcore.clients.clientbase import Client

class clientDAC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):

        super().__init__(
            args,
            id,
            train_samples,
            test_samples,
            train_slow=kwargs.get('train_slow', False),
            send_slow=kwargs.get('send_slow', False)
        )
        self.args = args

        self.critical_parameter = None    # flatten
        self.global_mask = None           # list of Tensors
        self.local_mask = None            # list of Tensors

        self.customized_model = copy.deepcopy(self.model)

        self.hist_sensitivity = None
        self.grad_momentum = None
        self.auxiliary_mask = None

    def train(self):

        trainloader = self.load_train_data()
        start_time = time.time()

        initial_model = copy.deepcopy(self.model)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                out = self.model(x)
                loss = self.loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.critical_parameter, self.global_mask, self.local_mask = \
            self.evaluate_critical_parameter(
                initial_model, self.model, tau=self.args.tau
            )

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module, tau: float):
        global_mask_list = []
        local_mask_list = []
        critical_param_list = []

        for (p_name, p_prev), (c_name, p_curr) in zip(prevModel.named_parameters(),
                                                      model.named_parameters()):
            g = p_curr.data - p_prev.data
            c = torch.abs(g * p_curr.data)

            flat_c = c.view(-1)
            num_params = flat_c.numel()

            nz = int(tau * num_params)
            if nz > 0:
                top_vals, _ = torch.topk(flat_c, nz)
                thresh = top_vals[-1].item()
                thresh = max(thresh, 1e-12)
            else:
                thresh = 1e12

            mask_local = (c >= thresh).int().cpu()
            mask_global = 1 - mask_local

            local_mask_list.append(mask_local)
            global_mask_list.append(mask_global)
            critical_param_list.append(mask_local.view(-1))

        critical_parameter = torch.cat(critical_param_list, dim=0)
        return critical_parameter, global_mask_list, local_mask_list

    def set_parameters(self, model: nn.Module):
        if self.local_mask is not None:
            index = 0
            for (name1, param1), (name2, param2), (name3, param3) in zip(
                    self.model.named_parameters(), model.named_parameters(),
                    self.customized_model.named_parameters()):
                param1.data = self.local_mask[index].to(self.device).float() * param3.data + \
                              self.global_mask[index].to(self.device).float() * param2.data
                index += 1
        else:
            super().set_parameters(model)

    def compute_distance(self, model1: nn.Module, model2: nn.Module):


        dist = 0.0
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            dist += (p1.data - p2.data).norm()**2
        return dist.item()