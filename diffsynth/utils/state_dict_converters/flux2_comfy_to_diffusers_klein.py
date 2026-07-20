import torch


def _iter_state_dict(sd):
    """Yield (key, tensor) from either a dict or a safetensors DiskMap."""
    # dict-like
    if hasattr(sd, "items"):
        for k, v in sd.items():
            yield k, v
        return
    # our DiskMap implements __iter__/__getitem__
    if hasattr(sd, "__iter__") and hasattr(sd, "__getitem__"):
        for k in sd:
            yield k, sd[k]
        return
    elif hasattr(sd, "keys"):
        for k in sd.keys():
            try:
                v = sd[k]
            except Exception:
                # Safetensors disk map
                v = sd.get_tensor(k)
            yield k, v
        return
    raise TypeError(f"Unsupported state_dict type: {type(sd)}")


def chunk_three(t):
    """Split a (3*D, ...) tensor into q/k/v."""
    d = t.shape[0] // 3
    return t[:d], t[d:2 * d], t[2 * d:]


def Flux2ComfyKleinToDiffusers(state_dict):
    """
    Convert Comfy/Flux2 single-file checkpoint (double_blocks/single_blocks, fused qkv)
    to DiffSynth Flux2DiT expected layout (split q/k/v, parallel single blocks).
    """
    out = {}

    for k, v in _iter_state_dict(state_dict):
        # Strip possible prefix
        if k.startswith("model.diffusion_model."):
            k = k[len("model.diffusion_model.") :]
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model.") :]

        # Time / guidance
        if k.startswith("time_in.in_layer."):
            out[k.replace("time_in.in_layer.", "time_guidance_embed.timestep_embedder.linear_1.")] = v
            continue
        if k.startswith("time_in.out_layer."):
            out[k.replace("time_in.out_layer.", "time_guidance_embed.timestep_embedder.linear_2.")] = v
            continue
        if k.startswith("img_in."):
            out[k.replace("img_in.", "x_embedder.")] = v
            continue
        if k.startswith("txt_in."):
            out[k.replace("txt_in.", "context_embedder.")] = v
            continue
        if k.startswith("double_stream_modulation_img.lin."):
            out[k.replace("double_stream_modulation_img.lin.", "double_stream_modulation_img.linear.")] = v
            continue
        if k.startswith("double_stream_modulation_txt.lin."):
            out[k.replace("double_stream_modulation_txt.lin.", "double_stream_modulation_txt.linear.")] = v
            continue
        if k.startswith("single_stream_modulation.lin."):
            out[k.replace("single_stream_modulation.lin.", "single_stream_modulation.linear.")] = v
            continue

        # Final layers
        if k.startswith("final_layer.adaLN_modulation.1."):
            out[k.replace("final_layer.adaLN_modulation.1.", "norm_out.linear.")] = v
            continue
        if k.startswith("final_layer.linear."):
            out[k.replace("final_layer.linear.", "proj_out.")] = v
            continue

        # Double blocks (image/text streams)
        if k.startswith("double_blocks."):
            parts = k.split(".")
            block_id = parts[1]
            rest = ".".join(parts[2:])

            prefix = f"transformer_blocks.{block_id}."

            # Attention img
            if rest.startswith("img_attn.qkv.weight"):
                q, k_, v_ = chunk_three(v)
                out[prefix + "attn.to_q.weight"] = q
                out[prefix + "attn.to_k.weight"] = k_
                out[prefix + "attn.to_v.weight"] = v_
                continue
            if rest.startswith("img_attn.qkv.lora_A.weight"):
                out[prefix + "attn.to_q.lora_A.weight"] = v
                out[prefix + "attn.to_k.lora_A.weight"] = v
                out[prefix + "attn.to_v.lora_A.weight"] = v
                continue
            if rest.startswith("img_attn.qkv.lora_B.weight"):
                q, k_, v_ = chunk_three(v)
                out[prefix + "attn.to_q.lora_B.weight"] = q
                out[prefix + "attn.to_k.lora_B.weight"] = k_
                out[prefix + "attn.to_v.lora_B.weight"] = v_
                continue
            if rest.startswith("img_attn.qkv.bias"):
                q, k_, v_ = chunk_three(v)
                out[prefix + "attn.to_q.bias"] = q
                out[prefix + "attn.to_k.bias"] = k_
                out[prefix + "attn.to_v.bias"] = v_
                continue
            if rest.startswith("img_attn.proj.weight"):
                out[prefix + "attn.to_out.0.weight"] = v
                continue
            if rest.startswith("img_attn.proj.lora_A.weight"):
                out[prefix + "attn.to_out.0.lora_A.weight"] = v
                continue
            if rest.startswith("img_attn.proj.lora_B.weight"):
                out[prefix + "attn.to_out.0.lora_B.weight"] = v
                continue
            if rest.startswith("img_attn.proj.bias"):
                out[prefix + "attn.to_out.0.bias"] = v
                continue
            if rest == "img_attn.norm.key_norm.scale":
                out[prefix + "attn.norm_k.weight"] = v
                continue
            if rest == "img_attn.norm.query_norm.scale":
                out[prefix + "attn.norm_q.weight"] = v
                continue

            # Attention txt (added kv)
            if rest.startswith("txt_attn.qkv.weight"):
                q, k_, v_ = chunk_three(v)
                out[prefix + "attn.add_q_proj.weight"] = q
                out[prefix + "attn.add_k_proj.weight"] = k_
                out[prefix + "attn.add_v_proj.weight"] = v_
                continue
            if rest.startswith("txt_attn.qkv.lora_A.weight"):
                out[prefix + "attn.add_q_proj.lora_A.weight"] = v
                out[prefix + "attn.add_k_proj.lora_A.weight"] = v
                out[prefix + "attn.add_v_proj.lora_A.weight"] = v
                continue
            if rest.startswith("txt_attn.qkv.lora_B.weight"):
                q, k_, v_ = chunk_three(v)
                out[prefix + "attn.add_q_proj.lora_B.weight"] = q
                out[prefix + "attn.add_k_proj.lora_B.weight"] = k_
                out[prefix + "attn.add_v_proj.lora_B.weight"] = v_
                continue
            if rest.startswith("txt_attn.qkv.bias"):
                q, k_, v_ = chunk_three(v)
                out[prefix + "attn.add_q_proj.bias"] = q
                out[prefix + "attn.add_k_proj.bias"] = k_
                out[prefix + "attn.add_v_proj.bias"] = v_
                continue
            if rest.startswith("txt_attn.proj.weight"):
                out[prefix + "attn.to_add_out.weight"] = v
                continue
            if rest.startswith("txt_attn.proj.lora_A.weight"):
                out[prefix + "attn.to_add_out.lora_A.weight"] = v
                continue
            if rest.startswith("txt_attn.proj.lora_B.weight"):
                out[prefix + "attn.to_add_out.lora_B.weight"] = v
                continue
            if rest.startswith("txt_attn.proj.bias"):
                out[prefix + "attn.to_add_out.bias"] = v
                continue
            if rest == "txt_attn.norm.key_norm.scale":
                out[prefix + "attn.norm_added_k.weight"] = v
                continue
            if rest == "txt_attn.norm.query_norm.scale":
                out[prefix + "attn.norm_added_q.weight"] = v
                continue

            # FF
            if rest.startswith("img_mlp.0."):
                out[prefix + "ff.linear_in." + rest.split(".", 2)[2]] = v
                continue
            if rest.startswith("img_mlp.2."):
                out[prefix + "ff.linear_out." + rest.split(".", 2)[2]] = v
                continue
            if rest.startswith("txt_mlp.0."):
                out[prefix + "ff_context.linear_in." + rest.split(".", 2)[2]] = v
                continue
            if rest.startswith("txt_mlp.2."):
                out[prefix + "ff_context.linear_out." + rest.split(".", 2)[2]] = v
                continue

            # Unknown double block entry
            continue

        # Single blocks (parallel attention)
        if k.startswith("single_blocks."):
            parts = k.split(".")
            block_id = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"single_transformer_blocks.{block_id}."

            if rest.startswith("linear1."):
                out[prefix + "attn.to_qkv_mlp_proj." + rest.split(".", 1)[1]] = v
                continue
            if rest.startswith("linear2."):
                out[prefix + "attn.to_out." + rest.split(".", 1)[1]] = v
                continue
            if rest == "norm.key_norm.scale":
                out[prefix + "attn.norm_k.weight"] = v
                continue
            if rest == "norm.query_norm.scale":
                out[prefix + "attn.norm_q.weight"] = v
                continue
            if rest.startswith("modulation.lin."):
                out[prefix + "norm.linear." + rest.split(".", 2)[2]] = v
                continue
            # other single-block params are not expected
            continue

        # Fallback: ignore unmatched keys

    return out
