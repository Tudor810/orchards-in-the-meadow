import torch

from agricultureai.training.checkpoints import (
    architecture_config_from_checkpoint,
    build_checkpoint_payload,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)


def test_self_describing_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 3)
    payload = build_checkpoint_payload(
        model,
        architecture_config={"model_type": "tiny", "seq_len": 45},
        training_metadata={"epoch": 7, "val_accuracy": 88.5},
    )
    path = tmp_path / "best.pt"
    torch.save(payload, path)

    loaded = load_checkpoint(path, map_location="cpu")
    cfg = architecture_config_from_checkpoint(loaded)
    state = model_state_dict_from_checkpoint(loaded)

    assert cfg["model_type"] == "tiny"
    assert cfg["seq_len"] == 45
    assert loaded["training_metadata"]["epoch"] == 7
    assert set(state.keys()) == set(model.state_dict().keys())
