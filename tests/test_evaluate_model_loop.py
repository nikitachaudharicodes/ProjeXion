from torch.utils.data import DataLoader, Dataset
from evaluation.evaluate import evaluate_model, compute_depth_metrics
from .mocks import DummyModel, DummyDataset, DummyLoss

def test_evaluate_model_loop():
    model = DummyModel()
    loader = DataLoader(DummyDataset(), batch_size=2)
    loss_fn = DummyLoss()

    val_loss, metrics = evaluate_model(model, loader, loss_fn)

    assert isinstance(val_loss, float)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["Abs Rel", "Sq Rel", "RMSE", "δ < 1.25", "δ < 1.25²", "δ < 1.25³"])
