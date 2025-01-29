import torch
import torch.optim as optim
import os
import datetime
from data_loader import train_loader, val_loader
from scripts.misc.customYOLO import CustomYOLO
from scripts.misc.utils import calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomYOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

os.makedirs("logs/checkpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    train_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    num_train_batches = len(train_loader)

    for images, bboxes, attr_labels, index_values, quad_values, ages, sexes in train_loader:
        images = images.to(device)
        bboxes = [b.to(device) for b in bboxes]
        attr_labels = [a.to(device) for a in attr_labels]
        index_values = [i.to(device) for i in index_values]
        quad_values = [q.to(device) for q in quad_values]

        optimizer.zero_grad()

        # raw model output is a tensor (not a list of Results objects)
        yolo_out, attributes_pred, index_pred, quad_pred = model(images)
        yolo_loss = 0.0  # placeholder if you haven't implemented a detection loss

        loss = model.compute_loss(
            yolo_loss=yolo_loss,
            attributes_pred=attributes_pred,
            attributes_true=attr_labels,
            index_pred=index_pred,
            index_true=index_values,
            quad_pred=quad_pred,
            quad_true=quad_values
        )
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        batch_metrics = calculate_metrics(index_pred, index_values, quad_pred, quad_values)
        for k in ["accuracy", "precision", "recall"]:
            train_metrics[k] += batch_metrics[k]

    for k in ["accuracy", "precision", "recall"]:
        train_metrics[k] /= num_train_batches

    avg_train_loss = total_train_loss / num_train_batches
    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"| Train Loss: {avg_train_loss:.4f} "
        f"| Acc: {train_metrics['accuracy']:.4f} "
        f"| Prec: {train_metrics['precision']:.4f} "
        f"| Recall: {train_metrics['recall']:.4f}"
    )

    model.eval()
    val_metrics = {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    num_val_batches = len(val_loader)

    with torch.no_grad():
        for images, bboxes, attr_labels, index_values, quad_values, ages, sexes in val_loader:
            images = images.to(device)
            bboxes = [b.to(device) for b in bboxes]
            attr_labels = [a.to(device) for a in attr_labels]
            index_values = [i.to(device) for i in index_values]
            quad_values = [q.to(device) for q in quad_values]

            yolo_out, attributes_pred, index_pred, quad_pred = model(images)
            yolo_loss = 0.0
            loss = model.compute_loss(
                yolo_loss=yolo_loss,
                attributes_pred=attributes_pred,
                attributes_true=attr_labels,
                index_pred=index_pred,
                index_true=index_values,
                quad_pred=quad_pred,
                quad_true=quad_values
            )
            val_metrics["loss"] += loss.item()

            batch_metrics = calculate_metrics(index_pred, index_values, quad_pred, quad_values)
            val_metrics["accuracy"] += batch_metrics["accuracy"]
            val_metrics["precision"] += batch_metrics["precision"]
            val_metrics["recall"] += batch_metrics["recall"]

    for k in ["accuracy", "precision", "recall"]:
        val_metrics[k] /= num_val_batches
    val_metrics["loss"] /= num_val_batches

    print(
        f"Validation | Loss: {val_metrics['loss']:.4f} "
        f"| Acc: {val_metrics['accuracy']:.4f} "
        f"| Prec: {val_metrics['precision']:.4f} "
        f"| Recall: {val_metrics['recall']:.4f}"
    )

end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_path = f"logs/checkpoints/weights-{end_time}.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved at {checkpoint_path}")
