def estimate_vllm_vram_from_config(
        config: dict, batch_size: int, max_model_len: int = None, dtype: str =None, param_tot: int = None, 
):
    mapping = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float": 4,
    }
    dtype = dtype or config.get("torch_dtype", "float16").lower()
    dtype_bytes = mapping.get(dtype, 2)

    max_pos = max_model_len or config["max_position_embeddings"] 
    vocab_size = config["vocab_size"]
    h = config["hidden_size"]
    h_inter = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]

    param_attn = num_layers * ( 4 * h**2 + 4 * h ) ## q k v comb with bias
    param_ffn  = num_layers * ( 2 * h * h_inter + 5 * h) ## h->h_inter->h with bias
    param_margin = 2 * vocab_size * h
    parm_lnorm = 2 * 2 * h ## attn and ffn, gamma and beta, elementwise on acitation [h]
    param_tot = param_tot or param_attn + param_ffn + parm_lnorm + param_margin

    model_weights = param_tot * dtype_bytes / 1024**3
    kv_cache = batch_size * max_pos * num_layers * h * 2 * dtype_bytes / 1024**3
    total_vram = model_weights + kv_cache
    print(f"model_weights = {model_weights:.3f} GB, kv_cache = {kv_cache:.3f} GB, total_vram = {total_vram:.3f} GB")
    return None


if __name__=="__main__":
    import json
    import os
    model_dirpath = "/mnt/disk2/xinghua.jia/workspace/finbot/resources/open_models/Qwen2.5-14B-Instruct"
    config_filepath = os.path.join(model_dirpath, "config.json")
    with open(config_filepath, "r") as f:
        config = json.load(f)

    estimate_vllm_vram_from_config(config, 1, 5000)

