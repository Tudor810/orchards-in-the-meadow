import torch

from agricultureai.models.breizhcrops_msresnet import BreizhCropsMSResNetExpert


def test_msresnet_expert_ignores_doy_when_disabled():
    model = BreizhCropsMSResNetExpert(
        num_classes=2,
        input_dim=10,
        hidden_dims=8,
        pretrained=False,
        use_doy=False,
    )
    x = torch.randn(2, 12, 10)
    doy = torch.randint(1, 366, (2, 12))
    out_no_doy = model(x)
    out_with_doy = model(x, doy=doy)
    assert out_no_doy.shape == (2, 2)
    torch.testing.assert_close(out_no_doy, out_with_doy)


def test_msresnet_expert_uses_doy_when_enabled():
    model = BreizhCropsMSResNetExpert(
        num_classes=2,
        input_dim=10,
        hidden_dims=8,
        pretrained=False,
        use_doy=True,
    )
    x = torch.randn(2, 12, 10)
    doy_a = torch.full((2, 12), 100, dtype=torch.long)
    doy_b = torch.full((2, 12), 200, dtype=torch.long)
    out_a = model(x, doy=doy_a)
    out_b = model(x, doy=doy_b)
    assert out_a.shape == (2, 2)
    assert not torch.allclose(out_a, out_b)
