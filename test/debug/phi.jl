using Transformers
using Transformers.Flux
using Transformers.HuggingFace
using Transformers.HuggingFace: HGFPhiPreTrainedModel, HGFPhiForCausalLM,  HGFLlamaPreTrainedModel, SelfAttention
using Transformers.HuggingFace: joinname, load_model
using Transformers.Layers: apply_on_namedtuple
using Transformers.HuggingFace: weighted_sum_mixing, gptneox_rope_multihead_qkv_attention, gptneox_rope_attention_score, generic_multihead_qkv_attention, gptneox_reorder
import Transformers.HuggingFace: one_init, zero_init, getweight
using Transformers.Layers: LayerNorm
using Transformers.NeuralAttentionlib: as_collapsed, _split_and_move_head,  generic_qkv_attention, mixing, attention_score, split_head, naive_qkv_attention, naive_attention_score, scaled_dot_product_score
using TextEncodeBase
using Statistics
using StatsBase
using Pickle
using Transformers.Flux.NNlib

function load_torch_matrix(filename)
	x = Pickle.Torch.THload(filename)
	x = Matrix(transpose(x[1,:,:]))
end

"""

 x: [bs, num_attention_heads, seq_len, head_size]
"""
function compare_tensors(x, filename)
	r = Pickle.Torch.THload(filename)
	size(r,1) !=1 && error("the first dimension should be one (one sample)")
	r = r[1,:,:,:]
	size(r,1) == size(x,3) || error("dimension mismatch")
    size(r,2) == size(x,2) || error("dimension mismatch")
    size(r,3) == size(x,1) || error("dimension mismatch")
	maximum(maximum(abs.(r[:,i,:] .- transpose(x[:,i,:]))) for i in 1:size(x,2))
end

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Phi
function  phi_rotary_embedding(dim, max_position_embeddings=2048, base=10000)
    inv_freq = 1 ./ (base .^ (collect(0:2:(dim-1)) ./ dim))
    inv_freq = vcat(inv_freq,inv_freq)
    t = 0:(max_position_embeddings-1)
    (sin.(inv_freq .* t'), cos.(inv_freq .* t'))
end


"""
	rotate_half(x)

	Rotates half the hidden dims of the input
"""
function rotate_half(x)
	d = size(x,1) ÷ 2
	x1 = @view x[1:d,:,:]
	x2 = @view x[d+1:end, :, :]
	cat(-x2, x1, dims = 1)
end

function apply_rotary_pos_emb(q, k, _cos::AbstractMatrix, _sin::AbstractMatrix)
    # _sin = reshape(_sin, size(_sin)..., 1)
    # _cos = reshape(_cos, size(_cos)..., 1)
    _sin = reshape(_sin, size(_sin)..., 1)
    _cos = reshape(_cos, size(_cos)..., 1)
	q_embed = q .* _cos .+ rotate_half(q) .* _sin
	k_embed = k .* _cos .+ rotate_half(k) .* _sin
	return (q_embed, k_embed)
end

model_type = model_name = "microsoft/phi-1"
cfg = Transformers.HuggingFace.load_config(model_type)
state_dict = Transformers.HuggingFace.load_state_dict(model_type; config=cfg)
state_dict = Dict(filter((kv) -> contains(kv[1], "model.layers.0."), collect(state_dict)))

# textenc = Transformers.HuggingFace.load_tokenizer(model_name)
# model = Transformers.HuggingFace.load_model(Transformers.HuggingFace.HGFPhiForCausalLM, cfg, state_dict, "")

s = "Tell me something about Julia?"

# input = encode(textenc, s).token 
# input = OneHotArray(OneHot{0x0000c477}.([ 24447, 503, 1224, 547, 22301, 31]))
# input_ref = Pickle.Torch.THload("/tmp/inputs.torch")["input_ids"]
# e = model.model.embed(input) # verify embedding
e_ref = load_torch_matrix("/tmp/embedding.torch")
e = e_ref

lprefix = "model.layers.0"
# residual = e.hidden_state
residual = e
ln = load_model(HGFPhiPreTrainedModel, Layers.LayerNorm, cfg, state_dict, joinname(lprefix, "input_layernorm"))
hidden_state = ln(residual)
hidden_state_ref = load_torch_matrix("/tmp/hidden_states.torch")


# this is where we want to do the self-attention. But it does not work, so 
# we need to learn, how to use it
sa = load_model(HGFPhiForCausalLM, SelfAttention, cfg, state_dict, joinname(lprefix, "self_attn"))
# attn_outputs = sa((;hidden_state = hidden_state_ref)).hidden_state
# attn_outputs .- load_torch_matrix("/tmp/attn_outputs.torch")

nt = (;hidden_state = hidden_state_ref)
qkv = apply_on_namedtuple(sa.qkv_proj, nt)
maximum(abs.(qkv.hidden_state[1] .- load_torch_matrix("/tmp/query_states.torch")))
maximum(abs.(qkv.hidden_state[2] .- load_torch_matrix("/tmp/key_states.torch")))
maximum(abs.(qkv.hidden_state[3] .- load_torch_matrix("/tmp/value_states.torch")))

# this part is about piercing the computation of attention scode
base, dim, head = 10000.0, 64, 32
hidden_size = 32
len = 6
_sincos = phi_rotary_embedding(32)
_sin = _sincos[1][:,1:len]
_cos = _sincos[2][:,1:len]
maximum(_sin .- Pickle.Torch.THload("/tmp/sin.torch")')
maximum(_cos .- Pickle.Torch.THload("/tmp/cos.torch")')

q,k,v = qkv.hidden_state
query_states = _split_and_move_head(head, as_collapsed(q))
key_states = _split_and_move_head(head, as_collapsed(k))
hv = _split_and_move_head(head, as_collapsed(v))

query_rot, query_pass = (
   query_states[1:32,:, :],	# sa.rotary_emb.dim = 32
   query_states[33:end, :, :],
)

compare_tensors(query_rot, "/tmp/query_rot.torch")
compare_tensors(query_pass, "/tmp/query_pass.torch")

key_rot, key_pass = (
   key_states[1:32,:, :],	# sa.rotary_emb.dim = 32
   key_states[33:end, :, :],
)

compare_tensors(key_rot, "/tmp/key_rot.torch")
compare_tensors(key_pass, "/tmp/key_pass.torch")


query_rot_pos, key_rot_pos = apply_rotary_pos_emb(query_rot, key_rot, _cos, _sin)

compare_tensors(query_rot_pos, "/tmp/query_rot_pos.torch")
compare_tensors(key_rot_pos, "/tmp/key_rot_pos.torch")

query_rot_states = cat(query_rot_pos, query_pass, dims=1)
key_rot_states = cat(key_rot_pos, key_pass, dims=1)

compare_tensors(query_rot_states, "/tmp/query_rot_states.torch")
compare_tensors(key_rot_states, "/tmp/key_rot_states.torch")


attn_weights = scaled_dot_product_score(query_rot_states, key_rot_states);
compare_tensors(attn_weights, "/tmp/attn_weights.torch")


# everything works until now

# attn_weights = attention_score(naive_attention_score(), query_rot_states, key_rot_states)


naive_attention_score()
generic_qkv_attention(weighted_sum_mixing, naive_attention_score(args...), q, k, v)
attn_output = generic_qkv_attention(weighted_sum_mixing, naive_attention_score(),query_rot_states, key_rot_states,  hv)

mixing(weighted_sum_mixing, as_collapsed(hv), naive_attention_score(), as_collapsed(query_rot_pos), as_collapsed(key_rot_pos))

attention_score(naive_attention_score(), as_collapsed(query_rot_states), as_collapsed(key_rot_states))
mixing(f, v, g, args...) = f(attention_score(g, args...), v)

r = Pickle.Torch.THload("/tmp/attn_output.torch")[1,:,:,:];
attn_output[:,:,1] .- r[:,1,:]'


function temp_softmax(logits; temperature=1.2)
    return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
    sorted = sort(probs, rev = true)
    indexes = partialsortperm(probs, 1:k, rev=true)
    index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
    return index
end

function generate_text(s=""; max_length=50)
    encoded = encode(textenc, s).token
    ids = encoded.onehots
    new_ids = ids[0:-1]
    ends_id = lookup(textenc.vocab, textenc.endsym)
    for i in 1:max_length
        input = (; token = OneHotArray(ids))
        outputs = model(input)
        logits = @view outputs.logit[:, end, 1]
        probs = temp_softmax(logits)
        new_id = top_k_sample(probs)[1]
        push!(ids, new_id)
        new_id == ends_id && break
    end
    return decode(textenc, OneHotArray(ids))
end

function generate(prompt, max_length)
    text_token = generate_text(prompt; max_length=max_length)
    gen_text = join(text_token)
    print("\n\nGenerated Text: ")
    println(gen_text)
end

s = """def print_prime(n):
   \"\"\"
   Print all primes between 1 and n
   \"\"\"

"""




textenc = Transformers.HuggingFace.load_tokenizer(model_name)
model = Transformers.HuggingFace.load_model(Transformers.HuggingFace.HGFPhiForCausalLM, cfg, state_dict, "")

prefix = "hello world"
tokens = TextEncodeBase.encode(textenc, "hello world")



model(tokens)




