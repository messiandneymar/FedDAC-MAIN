import time
import torch
import copy
import numpy as np
from flcore.servers.serverbase import Server
from flcore.clients.clientdac import clientDAC

class FedDAC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if not hasattr(args, 'num_clients') or args.num_clients <= 0:
            args.num_clients = 1

        self.clients = []
        for i in range(args.num_clients):
            train_samples = 100
            test_samples = 10
            c = clientDAC(args, i, train_samples, test_samples,
                          train_slow=False, send_slow=False)
            self.clients.append(c)
        self.current_num_join_clients = len(self.clients)

        self.epoch = -1
        self.Budget = []

    def select_clients(self):

        if len(self.clients) == 0:
            return []
        if self.current_num_join_clients <= 0 or \
           self.current_num_join_clients > len(self.clients):
            self.current_num_join_clients = len(self.clients)
        selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )
        return selected_clients

    def train(self):
        for round_i in range(self.global_rounds + 1):
            self.epoch = round_i
            start_t = time.time()

            self.selected_clients = self.select_clients()
            if len(self.selected_clients) == 0:
                print(f"Round {round_i}: no clients selected.")
                continue

            if round_i == 0:
                self.send_models()

            # 评估
            if round_i % self.eval_gap == 0:
                print(f"\n--- Round {round_i}, Evaluate ---")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            used_t = time.time() - start_t
            self.Budget.append(used_t)
            print(f"Round {round_i} finished, time cost: {used_t:.2f}s")

            if self.auto_break and \
               self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.rs_test_acc:
            print("\nBest accuracy =", max(self.rs_test_acc))
        if len(self.Budget) > 1:
            avg_t = sum(self.Budget[1:]) / len(self.Budget[1:])
            print("Average time cost:", f"{avg_t:.2f}s.")
        self.save_results()

    def compute_similarity(self, mask_i: torch.Tensor, mask_j: torch.Tensor):

        diff = torch.abs(mask_i - mask_j).sum().item()
        n = mask_i.numel()
        return diff / (2 * n)

    def aggregate_parameters(self):

        clients = self.selected_clients
        num_clients = len(clients)
        if num_clients == 0:
            return

        client_models = [copy.deepcopy(c.model.state_dict()) for c in clients]
        client_masks = []
        for c in clients:
            flatten_mask = torch.cat([m.view(-1) for m in c.local_mask], dim=0).cpu()
            client_masks.append(flatten_mask)

        S_mat = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(num_clients):
                S_mat[i, j] = self.compute_similarity(client_masks[i], client_masks[j])
        S_avg = torch.mean(S_mat).item()
        S_max = torch.max(S_mat).item()
        var_s = torch.var(S_mat).item()

        alpha = self.args.alpha
        t = self.epoch + 1
        time_factor = np.sqrt(t) / np.sqrt(alpha)
        S_threshold = 2

        alpha_agg = 0.6
        critical_agg_models = []
        for i in range(num_clients):
            Ci = []
            for j in range(num_clients):
                if S_mat[i, j].item() >= S_threshold:
                    Ci.append(j)

            ui_state = copy.deepcopy(client_models[i])
            for key in ui_state.keys():
                param_acc = client_models[i][key].clone()
                for j in Ci:
                    param_acc += client_models[j][key]
                param_avg = param_acc / (len(Ci) + 1) if len(Ci) > 0 else param_acc

                ui_state[key] = alpha_agg * client_models[i][key] + \
                                (1 - alpha_agg) * param_avg
            critical_agg_models.append(ui_state)

        final_personalized = []
        epsilon = 1e-6
        for i in range(num_clients):
            wi_star = copy.deepcopy(client_models[i])
            distances = []
            for j in range(num_clients):
                d_ij = 0.0
                for pkey in wi_star.keys():
                    param_i = wi_star[pkey]
                    param_j = client_models[j][pkey]
                    if torch.is_floating_point(param_i) and torch.is_floating_point(param_j):
                        d_ij += (param_i - param_j).norm() ** 2
                distances.append(d_ij + epsilon)

            inv_dist = [1.0 / d for d in distances]
            sum_inv = sum(inv_dist)
            pis = [v / sum_inv for v in inv_dist]
            new_state = {}
            for key in critical_agg_models[i].keys():
                local_param = critical_agg_models[i][key]
                aggregated_param = torch.zeros_like(local_param)
                for j in range(num_clients):
                    aggregated_param += pis[j] * client_models[j][key]

                k_percent = 1
                k = int(local_param.numel() * k_percent)
                _, top_indices = torch.topk(local_param.abs().view(-1), k)

                mask = torch.zeros_like(local_param, dtype=torch.bool)
                mask.view(-1)[top_indices] = True

                new_state[key] = torch.where(
                    mask,
                    local_param,
                    aggregated_param
                )
            final_personalized.append(new_state)

        for i in range(num_clients):
            clients[i].customized_model.load_state_dict(final_personalized[i])

    def send_models(self):
        for client in self.selected_clients:
            if client.customized_model is not None:
                client.set_parameters(client.customized_model)
            else:
                client.set_parameters(self.global_model)