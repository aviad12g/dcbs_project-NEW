import copy
from messages_completion import MessageCompleter, HuggingFaceModelInterface

EXAMPLE_CONVOS = [
    [{"role":"system","content":"You are terse."},
     {"role":"user","content":"2+2?"}],
    [{"role":"system","content":"You are terse."},
     {"role":"user","content":"Capital of France?"}],
]

def test_batch_invariance():
    model = HuggingFaceModelInterface("meta-llama/Meta-Llama-3-8B-Instruct")
    model.set_seed(42)
    comp = MessageCompleter(model, max_new_tokens=8)
    
    seq_ids = []
    for convo in EXAMPLE_CONVOS:
        out = comp.complete([copy.deepcopy(convo)], use_batching=False, return_logprobs=False)
        seq_ids.append(out.token_ids if hasattr(out, "token_ids") else out.completions[0].token_ids)
    
    batched = comp.complete(copy.deepcopy(EXAMPLE_CONVOS), use_batching=True, return_logprobs=False)
    batched_ids = [c.token_ids for c in batched.completions]
    
    assert seq_ids == batched_ids