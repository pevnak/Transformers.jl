# I let julia handle conda and stuff there
# add PythonCall CondaPkg
# ] conda add pytorch
# ] conda add math
# /private/tmp/.CondaPkg/env/bin/pip install git+https://github.com/huggingface/transformers
import torch
import transformers
import math
torch.manual_seed(0)

def rotate_half(x):
   """Rotates half the hidden dims of the input."""
   x1 = x[..., : x.shape[-1] // 2]
   x2 = x[..., x.shape[-1] // 2 :]
   return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
   cos = cos[position_ids].unsqueeze(unsqueeze_dim)
   sin = sin[position_ids].unsqueeze(unsqueeze_dim)
   q_embed = (q * cos) + (rotate_half(q) * sin)
   k_embed = (k * cos) + (rotate_half(k) * sin)
   return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv with llama->phi

def repeat_kv(hidden_states, n_rep):
   """
   This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
   num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
   """
   batch, num_key_value_heads, slen, head_dim = hidden_states.shape
   if n_rep == 1:
      return hidden_states
   hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
   return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


model =transformers.AutoModelForCausalLM.from_pretrained('microsoft/phi-1', torch_dtype=torch.float32, trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/phi-1', trust_remote_code=True)

s = 'Tell me something about Julia?'
inputs = tokenizer(s, return_tensors='pt', return_attention_mask=False)
torch.save(inputs, '/tmp/inputs.torch')
e = model.model.embed_tokens(inputs.input_ids)
l = model.model.layers[0]
torch.save(e, '/tmp/embedding.torch')
hidden_states = l.input_layernorm(e)
torch.save(hidden_states, '/tmp/hidden_states.torch')


attn_outputs, self_attn_weights, present_key_value = l.self_attn(hidden_states)
torch.save(attn_outputs, '/tmp/attn_outputs.torch')

sa = l.self_attn
query_states = sa.q_proj(hidden_states)
torch.save(query_states, '/tmp/query_states.torch')
key_states = sa.k_proj(hidden_states)
torch.save(key_states, '/tmp/key_states.torch')
value_states = sa.v_proj(hidden_states)
torch.save(value_states, '/tmp/value_states.torch')


bsz, q_len, _ = hidden_states.size()
query_states = query_states.view(bsz, q_len, sa.num_heads, sa.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, sa.num_key_value_heads, sa.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, sa.num_key_value_heads, sa.head_dim).transpose(1, 2)

kv_seq_len = key_states.shape[-2]
cos, sin = sa.rotary_emb(value_states, seq_len=kv_seq_len)
torch.save(cos, '/tmp/cos.torch')
torch.save(sin, '/tmp/sin.torch')

# Partial rotary embedding
query_rot, query_pass = (
   query_states[..., : sa.rotary_emb.dim],
   query_states[..., sa.rotary_emb.dim :],
)

torch.save(query_rot, '/tmp/query_rot.torch')
torch.save(query_pass, '/tmp/query_pass.torch')

key_rot, key_pass = (
   key_states[..., : sa.rotary_emb.dim],
   key_states[..., sa.rotary_emb.dim :],
)

torch.save(key_rot, '/tmp/key_rot.torch')
torch.save(key_pass, '/tmp/key_pass.torch')

# [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
query_rot_pos, key_rot_pos = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, None)
torch.save(query_rot_pos, '/tmp/query_rot_pos.torch')
torch.save(key_rot_pos, '/tmp/key_rot_pos.torch')


# [batch_size, seq_length, num_heads, head_dim]
query_rot_states = torch.cat((query_rot_pos, query_pass), dim=-1)
key_rot_states = torch.cat((key_rot_pos, key_pass), dim=-1)

torch.save(query_rot_states, '/tmp/query_rot_states.torch')
torch.save(key_rot_states, '/tmp/key_rot_states.torch')


key_states = repeat_kv(key_states, sa.num_key_value_groups)
value_states = repeat_kv(value_states, sa.num_key_value_groups)

# Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
key_rot_states.size()
query_rot_states.size()
attn_weights = torch.matmul(
   query_rot_states.to(torch.float32), key_rot_states.to(torch.float32).transpose(2, 3)
)/ math.sqrt(sa.head_dim)
attn_weights.size()

torch.save(attn_weights, '/tmp/attn_weights.torch')

# upcast attention to fp32
attn_weights_softmax = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
torch.save(attn_weights_softmax, '/tmp/attn_weights_softmax.torch')

attn_output = torch.matmul(attn_weights_softmax, value_states)
torch.save(attn_output, '/tmp/attn_output.torch')

if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
   raise ValueError(
       f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
       f" {attn_output.size()}"
   )

attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

attn_output = self.dense(attn_output)

if not output_attentions:
   attn_weights = None

return attn_output, attn_weights, past_key_value


feed_forward_hidden_states = l.mlp(hidden_states)
torch.save(feed_forward_hidden_states, '/tmp/feed_forward_hidden_states.torch')
output = attn_outputs + feed_forward_hidden_states + e
torch.save(output, '/tmp/output.torch')

