from torch import nn
from transformers import RobertaForSequenceClassification

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, hidden_size)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # 加入残差连接以保留输入信息

class RobertaForSequenceClassificationWithAdapters(RobertaForSequenceClassification):
    def __init__(self, config, adapter_dim=64):
        super().__init__(config)
        self.adapter_dim = adapter_dim
        
        # 在嵌入层后添加 Adapter 层
        self.embedding_adapter = Adapter(config.hidden_size, adapter_dim)
        
        # 在每个 Transformer 层的前馈网络（FFN）前后添加 Adapter 层
        for layer in self.roberta.encoder.layer:
            layer.output.adapter_before_ffn = Adapter(config.hidden_size, adapter_dim)
            layer.output.adapter_after_ffn = Adapter(config.hidden_size, adapter_dim)

        # 冻结原始 RoBERTa 模型的所有参数，仅允许 Adapter 和分类头的参数更新
        self._freeze_model_parameters()

    def _freeze_model_parameters(self):
        # 冻结 RoBERTa 模型的所有参数
        for param in self.roberta.parameters():
            param.requires_grad = False
        # 允许 Adapter 层和分类头的参数更新
        for layer in self.roberta.encoder.layer:
            for param in layer.output.adapter_before_ffn.parameters():
                param.requires_grad = True
            for param in layer.output.adapter_after_ffn.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.embedding_adapter.parameters():
            param.requires_grad = True
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        # 获取嵌入层输出并应用 Adapter
        hidden_states = self.embedding_adapter(self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        ))

        # 扩展注意力掩码
        extended_attention_mask = self.roberta.get_extended_attention_mask(attention_mask, input_ids.shape, input_ids.device)

        # 逐层应用自注意力、Adapter 和 FFN
        for layer in self.roberta.encoder.layer:
            # 自注意力机制
            attention_output = layer.attention(
                hidden_states, 
                attention_mask=extended_attention_mask, 
                head_mask=head_mask[layer] if head_mask is not None else None
            )[0]
            
            # Adapter、FFN 和层归一化处理
            attention_output = layer.output.adapter_before_ffn(attention_output)
            ffn_output = layer.output.LayerNorm(
                layer.output.dense(layer.intermediate.intermediate_act_fn(
                    layer.intermediate.dense(attention_output)
                )) + attention_output
            )
            
            # 在 FFN 后应用 Adapter
            hidden_states = layer.output.adapter_after_ffn(ffn_output)

        # 分类头使用 [CLS] token 表示
        logits = self.classifier(hidden_states)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output={
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states
        }
        
        return output