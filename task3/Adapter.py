from torch import nn
from transformers import RobertaForSequenceClassification

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim=64):
        super(Adapter, self).__init__()
        # Down-project to a smaller dimensional space
        self.down_project = nn.Linear(hidden_size, adapter_dim)
        self.activation = nn.ReLU()
        # Up-project back to the original hidden size
        self.up_project = nn.Linear(adapter_dim, hidden_size)

    def forward(self, x):
        residual = x  # Save the input for residual connection
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Add residual connection to preserve input information

class RobertaForSequenceClassificationWithAdapters(RobertaForSequenceClassification):
    def __init__(self, config, adapter_dim=64):
        super().__init__(config)
        self.adapter_dim = adapter_dim
        
        # Add an Adapter layer after the embedding layer
        self.embedding_adapter = Adapter(config.hidden_size, adapter_dim)
        
        # Add Adapter layers before and after the feed-forward network (FFN) in each Transformer layer
        for layer in self.roberta.encoder.layer:
            layer.output.adapter_before_ffn = Adapter(config.hidden_size, adapter_dim)
            layer.output.adapter_after_ffn = Adapter(config.hidden_size, adapter_dim)

        # Freeze all parameters of the original RoBERTa model, allowing only Adapters and classification head to update
        self._freeze_model_parameters()

    def _freeze_model_parameters(self):
        # Freeze all parameters of the RoBERTa model
        for param in self.roberta.parameters():
            param.requires_grad = False
        # Allow updating only for Adapter layers and the classification head
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
        # Pass input through embedding layer and apply Adapter
        hidden_states = self.embedding_adapter(self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        ))

        # Extend the attention mask to apply across all layers
        extended_attention_mask = self.roberta.get_extended_attention_mask(attention_mask, input_ids.shape, input_ids.device)

        # Apply self-attention, Adapter, and FFN in each layer
        for layer in self.roberta.encoder.layer:
            # Self-attention mechanism
            attention_output = layer.attention(
                hidden_states, 
                attention_mask=extended_attention_mask, 
                head_mask=head_mask[layer] if head_mask is not None else None
            )[0]
            
            # Adapter, FFN, and layer normalization
            attention_output = layer.output.adapter_before_ffn(attention_output)
            ffn_output = layer.output.LayerNorm(
                layer.output.dense(layer.intermediate.intermediate_act_fn(
                    layer.intermediate.dense(attention_output)
                )) + attention_output
            )
            
            # Apply Adapter after FFN
            hidden_states = layer.output.adapter_after_ffn(ffn_output)

        # Classification head uses the [CLS] token representation
        logits = self.classifier(hidden_states)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states
        }
        
        return output
