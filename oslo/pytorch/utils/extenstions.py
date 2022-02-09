from functools import partial
from typing import Optional


def restrict_embedding_resizing(model):
    def resize_token_embeddings(new_num_tokens: Optional[int] = None, **kwargs):
        raise RuntimeError(
            "you can't use ``model.resize_token_embeddings()`` if you initialized OSLO.\n"
            "please resize token embedding size before OSLO initialization."
        )

    setattr(
        model, "resize_token_embeddings", partial(resize_token_embeddings, self=model)
    )

    return model
