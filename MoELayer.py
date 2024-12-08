from transformers import BertForMaskedLM
from transformers import TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
temp_indices = []

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicRouter(nn.Module):
    def __init__(self, input_dim,
                 num_experts=8,
                 top_k=2,
                 noise_std=0.1):
        super(DynamicRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(input_dim, num_experts)
        self.noise_std = noise_std
    def forward(self, x):
        logits = self.linear(x)
        noise = torch.randn_like(logits) * self.noise_std
        logits += noise
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

        zeros = torch.full_like(logits, float('-inf')).to(logits.device)
        sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
        global temp_indices
        temp_indices.append([logits, top_k_indices])
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, top_k_indices


class SparseMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=8, top_k=2, dropout=0.15):
        super(SparseMoE, self).__init__()
        self.router = DynamicRouter(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, dropout) for _ in range(num_experts)])
        self.shared_expert = Expert(input_dim, output_dim, dropout)
        self.top_k = top_k
        self.beta = nn.Parameter(torch.tensor(0.7), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad=True)

    def forward(self, x):
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(x)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                final_output[expert_mask] += weighted_output.squeeze(1)

        weights = F.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        a, b = weights[0], weights[1]
        final_output = self.shared_expert(x) * a + final_output * b
        # global temp_indices
        # temp_indices.append([final_output, indices])
        return final_output


class SparseMoEFFN(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2, dropout=0.15):
        super(SparseMoEFFN, self).__init__()
        self.sparse_moe = SparseMoE(input_dim=768,
                                    output_dim=3072,
                                    num_experts=num_experts,
                                    top_k=top_k,
                                    dropout=dropout)

    def forward(self, x):
        return self.sparse_moe(x)


class BertWwmMoE(BertForMaskedLM):
    def __init__(self, config, num_experts=8, top_k=2, dropout=0.05):
        super(BertWwmMoE, self).__init__(config)
        for index, layer in enumerate(self.bert.encoder.layer):
            if 8 <= index <= 15:
                if index % 2 == 1:
                    continue
                moe_ffn = SparseMoEFFN(config=config,
                                       num_experts=8,
                                       top_k=2,
                                       dropout=dropout)

                for index, expert in enumerate(moe_ffn.sparse_moe.shared_expert.net):
                    if index == 0:
                        expert.weight.data = layer.intermediate.dense.weight.data.clone()
                        expert.bias.data = layer.intermediate.dense.bias.data.clone()
                    if index == 3:
                        expert.weight.data = layer.output.dense.weight.data.clone()
                        expert.bias.data = layer.output.dense.bias.data.clone()

                layer.intermediate.dense = moe_ffn
                layer.output.dense = nn.Identity()

            if 5 <= index <= 7:
                if index % 2 == 1:
                    continue
                moe_ffn = SparseMoEFFN(config=config,
                                       num_experts=4,
                                       top_k=1,
                                       dropout=dropout)

                for index, expert in enumerate(moe_ffn.sparse_moe.shared_expert.net):
                    if index == 0:
                        expert.weight.data = layer.intermediate.dense.weight.data.clone()
                        expert.bias.data = layer.intermediate.dense.bias.data.clone()
                    if index == 3:
                        expert.weight.data = layer.output.dense.weight.data.clone()
                        expert.bias.data = layer.output.dense.bias.data.clone()

                layer.intermediate.dense = moe_ffn
                layer.output.dense = nn.Identity()
            if 0 <= index <= 4:
                if index % 2 == 1:
                    continue
                moe_ffn = SparseMoEFFN(config=config,
                                       num_experts=2,
                                       top_k=1,
                                       dropout=0.1)

                for index, expert in enumerate(moe_ffn.sparse_moe.shared_expert.net):
                    if index == 0:
                        expert.weight.data = layer.intermediate.dense.weight.data.clone()
                        expert.bias.data = layer.intermediate.dense.bias.data.clone()
                    if index == 3:
                        expert.weight.data = layer.output.dense.weight.data.clone()
                        expert.bias.data = layer.output.dense.bias.data.clone()

                layer.intermediate.dense = moe_ffn
                layer.output.dense = nn.Identity()
            if 0 <= index <= 0:
                if index % 2 == 1:
                    continue
                moe_ffn = SparseMoEFFN(config=config,
                                       num_experts=2,
                                       top_k=1,
                                       dropout=0.1)

                for index, expert in enumerate(moe_ffn.sparse_moe.shared_expert.net):
                    if index == 0:
                        expert.weight.data = layer.intermediate.dense.weight.data.clone()
                        expert.bias.data = layer.intermediate.dense.bias.data.clone()
                    if index == 3:
                        expert.weight.data = layer.output.dense.weight.data.clone()
                        expert.bias.data = layer.output.dense.bias.data.clone()

                layer.intermediate.dense = moe_ffn
                layer.output.dense = nn.Identity()
class EvaluationCallback(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_evaluate(self, args, state, control, **kwargs):
        # 获取并打印指定层的权重
        for index, layer in enumerate(self.model.bert.encoder.layer):
            if 0 <= index < 6:
                print(layer.intermediate.dense.sparse_moe.alpha.data)
                print(layer.intermediate.dense.sparse_moe.beta.data)
