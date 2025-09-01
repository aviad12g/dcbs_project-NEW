import pytest


class DummyModel:
    """Minimal model to exercise DCBS generate path deterministically."""

    def __init__(self, logits_seq):
        # logits_seq: list of tensors [B, V] yielded per step
        self._logits_seq = list(logits_seq)
        self._cursor = 0
        self._device = "cpu"
        self.tokenizer = type("T", (), {"eos_token_id": 0})()

    @property
    def model_name(self):
        return "dummy"

    @property
    def device(self):
        return self._device

    def get_embedding_layer(self):
        import torch
        # Return a fake embedding layer with weight to provide device
        emb = torch.nn.Embedding(10, 4)
        return emb

    def next_token_logits(self, inputs):
        out = self._logits_seq[self._cursor]
        self._cursor += 1
        return out

    def append_tokens(self, inputs, new_token_ids):
        return inputs

    def detokenize(self, ids):
        return " ".join(str(x) for x in ids)


def test_dcbs_greedy_tiebreak(monkeypatch):
    import torch
    from messages_completion.samplers.dcbs import DCBSSampler as Wrapper

    # Craft logits with a tie between two tokens; DCBS sample_batch uses argmax (first index wins)
    logits = torch.tensor([[0.0, 1.0, 1.0, 0.5]])  # B=1, V=4
    model = DummyModel([logits])

    # Monkeypatch internal DCBS sampler to select argmax over logits directly
    w = Wrapper(k=4, top_n=4)
    class FakeCore:
        def sample_batch(self, lb, filter_tokens_batch=None, context=None):
            # choose first max index (stable tie-break => lowest token id among maxima)
            return lb.argmax(dim=-1).tolist()
    w._sampler = FakeCore()

    token_seqs, _ = w.generate(model, {"input_ids": torch.zeros((1, 1), dtype=torch.long)}, max_new_tokens=1, return_logprobs=False)
    assert token_seqs == [[1]]  # 1 < 2 in tie


def test_dcbs_batch_invariance(monkeypatch):
    import torch
    from messages_completion.samplers.dcbs import DCBSSampler as Wrapper

    # Two-step deterministic logits: step1 identical, step2 identical
    step1 = torch.tensor([[0.1, 0.9], [0.1, 0.9]])  # B=2
    step2 = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
    model_batch = DummyModel([step1, step2])

    w = Wrapper(k=2, top_n=2)
    class FakeCore:
        def sample_batch(self, lb, filter_tokens_batch=None, context=None):
            return lb.argmax(dim=-1).tolist()
    w._sampler = FakeCore()

    # Batched
    tokens_b, _ = w.generate(model_batch, {"input_ids": torch.zeros((2, 1), dtype=torch.long)}, max_new_tokens=2, return_logprobs=False)

    # Sequential (two separate single-batch runs using same logits per step)
    model_seq1 = DummyModel([step1[:1], step2[:1]])
    t1, _ = w.generate(model_seq1, {"input_ids": torch.zeros((1, 1), dtype=torch.long)}, max_new_tokens=2, return_logprobs=False)
    model_seq2 = DummyModel([step1[1:], step2[1:]])
    t2, _ = w.generate(model_seq2, {"input_ids": torch.zeros((1, 1), dtype=torch.long)}, max_new_tokens=2, return_logprobs=False)

    assert tokens_b == [t1[0], t2[0]]

