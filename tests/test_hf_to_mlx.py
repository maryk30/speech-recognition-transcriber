import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from hf_to_mlx import convert_state_dict  # noqa: E402


def test_name_mapping_and_conv_transpose():
    hf = {
        "model.encoder.conv1.weight": np.zeros((768, 80, 3)),
        "model.encoder.layers.3.self_attn.q_proj.weight": np.ones((4, 4)),
        "model.encoder.layers.3.self_attn.k_proj.weight": np.ones((4, 4)),
        "model.encoder.layers.3.final_layer_norm.bias": np.ones(4),
        "model.encoder.layer_norm.weight": np.ones(4),
        "model.encoder.embed_positions.weight": np.ones((1500, 4)),
        "model.decoder.layers.0.encoder_attn.out_proj.bias": np.ones(4),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": np.ones(4),
        "model.decoder.layers.11.fc1.weight": np.ones((4, 4)),
        "model.decoder.embed_tokens.weight": np.ones((10, 4)),
        "model.decoder.embed_positions.weight": np.ones((448, 4)),
        "model.decoder.layer_norm.bias": np.ones(4),
        "proj_out.weight": np.ones((10, 4)),
    }
    out = convert_state_dict(hf)
    assert out["encoder.conv1.weight"].shape == (768, 3, 80)
    assert {
        "encoder.blocks.3.attn.query.weight", "encoder.blocks.3.attn.key.weight",
        "encoder.blocks.3.mlp_ln.bias", "encoder.ln_post.weight",
        "decoder.blocks.0.cross_attn.out.bias", "decoder.blocks.0.cross_attn_ln.weight",
        "decoder.blocks.11.mlp1.weight", "decoder.token_embedding.weight",
        "decoder.positional_embedding", "decoder.ln.bias",
    } <= set(out)
    assert not any("embed_positions" in k and k.startswith("encoder") for k in out)   # sinusoid, not exported
    assert "proj_out.weight" not in out                                               # tied to token embedding
