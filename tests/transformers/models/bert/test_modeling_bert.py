import torch
from oslo.transformers.models.bert.modeling_bert import BertModel

try:
    from transformers import AutoConfig
    from transformers.models.bert.modeling_bert import (
        BertModel as TransformersBertModel,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


def gradient_check(
    oslo_model,
    transformers_model,
    batch_size=8,
    seq_length=10,
):
    def forward_check(oslo_result, orig_result, atol=1e-5):
        three_multiple = lambda x, y, z: x * y * z
        result = torch.isclose(oslo_result, orig_result, atol=atol).sum()
        num_elements = three_multiple(*oslo_result.size())
        print(f"forward_same_ratio: {result/num_elements:.4f}\n")

    print("\n======================= Gradient Check =======================\n")
    batch_size, seq_length = batch_size, seq_length
    sample = torch.randint(0, 1000, (batch_size, seq_length))

    oslo_model.eval()
    transformers_model.eval()

    oslo_result = oslo_model(sample).last_hidden_state
    orig_result = transformers_model(sample).last_hidden_state

    forward_check(oslo_result, orig_result)

    oslo_result.sum().backward()
    orig_result.sum().backward()

    multiple = lambda x, y: x * y
    for oslo, orig in zip(
        oslo_model.named_parameters(), transformers_model.named_parameters()
    ):
        oslo_name, oslo_param = oslo
        orig_name, orig_param = orig

        if oslo_param.grad is None and orig_param.grad is None:
            continue
        if oslo_param.grad.dim() == 2:
            num_params = multiple(*oslo_param.grad.size())
        else:
            num_params = len(oslo_param.grad)

        if oslo_name == orig_name:
            result = torch.isclose(oslo_param.grad, orig_param.grad).sum()
            print(
                f"{oslo_name:50s} same_grad_ratio:  {result/num_params:.4f}   num_params:{num_params:9d}   num_same_grad:{result:9d}"
            )

        oslo_model.zero_grad()
        transformers_model.zero_grad()
    print("\n============================ End ============================\n")


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.hidden_act = "gelu_new"
    oslo_model = BertModel.from_pretrained("bert-base-uncased")
    orig_model = TransformersBertModel.from_pretrained(
        "bert-base-uncased", config=config
    )
    gradient_check(oslo_model, orig_model)
