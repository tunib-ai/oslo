import torch
from oslo.transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
)

try:
    from transformers import (
        GPT2Model as TransformersGPT2Model,
        GPT2LMHeadModel as TransformersGPT2LMHeadModel,
        GPT2ForSequenceClassification as TransformersGPT2ForSequenceClassification,
        GPT2ForTokenClassification as TransformersGPT2ForTokenClassification,
        AutoConfig,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


def gradient_check(
    oslo_model, transformers_model, batch_size=8, seq_length=10, return_logits=True
):
    print("\n======================= Gradient Check =======================\n")
    batch_size, seq_length = batch_size, seq_length
    sample = torch.randint(0, 1000, (batch_size, seq_length))

    oslo_model.eval()
    transformers_model.eval()

    if return_logits:
        oslo_result = oslo_model(sample).logits
        orig_result = transformers_model(sample).logits
    else:
        oslo_result = oslo_model(sample).last_hidden_state
        orig_result = transformers_model(sample).last_hidden_state

    if torch.allclose(oslo_result, orig_result, atol=1e-3):
        print("Forward result is same\n")
    else:
        print("Forward result is different\n")

    oslo_result.sum().backward()
    orig_result.sum().backward()

    multiple = lambda x, y: x * y
    for oslo, orig in zip(
        oslo_model.named_parameters(), transformers_model.named_parameters()
    ):
        oslo_name, oslo_param = oslo
        orig_name, orig_param = orig

        if oslo_param.grad.dim() == 2:
            num_params = multiple(*oslo_param.grad.size())
        else:
            num_params = len(oslo_param.grad)

        if oslo_name == orig_name:
            result = torch.isclose(oslo_param.grad, orig_param.grad, atol=1e-5).sum()
            if return_logits:
                print(
                    f"{oslo_name:36s} true_grad_ratio: {result/num_params:.4f}   num_param: {num_params:8d}   num_true_grad: {result:8d}"
                )
            else:
                print(
                    f"{oslo_name:24s} true_grad_ratio: {result/num_params:.4f}   num_param: {num_params:8d}   num_true_grad: {result:8d}"
                )

    oslo_model.zero_grad()
    transformers_model.zero_grad()
    print("\n============================ End ============================\n")


if __name__ == "__main__":
    oslo_model = GPT2Model.from_pretrained("gpt2")
    orig_model = TransformersGPT2Model.from_pretrained("gpt2")
    gradient_check(oslo_model, orig_model, return_logits=False)

    oslo_model = GPT2LMHeadModel.from_pretrained("gpt2")
    orig_model = TransformersGPT2LMHeadModel.from_pretrained("gpt2")
    gradient_check(oslo_model, orig_model)

    oslo_model = GPT2ForSequenceClassification.from_pretrained("gpt2")
    orig_model = TransformersGPT2ForSequenceClassification.from_pretrained("gpt2")
    gradient_check(oslo_model, orig_model, batch_size=1)

    oslo_model = GPT2ForTokenClassification.from_pretrained("gpt2")
    orig_model = TransformersGPT2ForTokenClassification.from_pretrained("gpt2")
    gradient_check(oslo_model, orig_model)

    config = AutoConfig.from_pretrained("gpt2")
    config.reorder_and_upcast_attn = True
    oslo_model = GPT2Model.from_pretrained("gpt2", config=config)
    orig_model = TransformersGPT2Model.from_pretrained("gpt2", config=config)
    gradient_check(oslo_model, orig_model, return_logits=False)
