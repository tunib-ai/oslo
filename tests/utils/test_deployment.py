from transformers import GPT2Tokenizer

from oslo import GPT2LMHeadModel, torch


def test_inference(fp16):
    model_3d = GPT2LMHeadModel.from_pretrained_with_parallel(
        "gpt2",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        deployment=True,
    ).eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if fp16:
        model_1d = GPT2LMHeadModel.from_pretrained("gpt2").half().eval().cuda()
    else:
        model_1d = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()

    batch_encoding = tokenizer(text="Hello I am Kevin. Today,", return_tensors="pt").to(
        "cuda"
    )

    output_3d = model_3d.generate(**batch_encoding, num_beams=4, no_repeat_ngram_size=3)
    output_1d = model_1d.generate(**batch_encoding, num_beams=4, no_repeat_ngram_size=3)

    print(
        f"--fp16:{fp16}\n"
        f"--test result: \n1D:{tokenizer.decode(output_1d[0])}\n2D:{tokenizer.decode(output_3d[0])}\n"
    )

    del model_3d
    del model_1d


if __name__ == "__main__":
    test_inference(fp16=True)
