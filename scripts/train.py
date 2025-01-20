import torch
import torch.optim as optim
import os
import datetime
from data_loader import train_loader, val_loader
from customYOLO import CustomYOLO
from utils import calculate_metrics

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomYOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# Ensure checkpoint directory exists
os.makedirs("logs/checkpoints", exist_ok=True)

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    num_batches = len(train_loader)

    for images, bboxes, attr_labels, index_values, quad_values, ages, sexes in train_loader:
        images = images.to(device)
        attr_labels = attr_labels.to(device)
        index_values = index_values.to(device)
        quad_values = quad_values.to(device)

        optimizer.zero_grad()
        yolo_out, attributes_pred, index_pred, quad_pred = model(images)

        loss = model.compute_loss(
            yolo_loss=yolo_out.loss,
            attributes_pred=attributes_pred,
            attributes_true=attr_labels,
            index_pred=index_pred,
            index_true=index_values,
            quad_pred=quad_pred,
            quad_true=quad_values
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        batch_metrics = calculate_metrics(attributes_pred, attr_labels, index_pred, index_values, quad_pred, quad_values)
        for key in train_metrics:
            train_metrics[key] += batch_metrics[key]

    for key in train_metrics:
        train_metrics[key] /= num_batches

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss:.4f} | Accuracy: {train_metrics['accuracy']:.4f} | Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | F1 Score: {train_metrics['f1_score']:.4f}")

    # Validation
    model.eval()
    val_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    num_batches = len(val_loader)
    with torch.no_grad():
        for images, bboxes, attr_labels, index_values, quad_values, ages, sexes in val_loader:
            images = images.to(device)
            attr_labels = attr_labels.to(device)
            index_values = index_values.to(device)
            quad_values = quad_values.to(device)

            yolo_out, attributes_pred, index_pred, quad_pred = model(images)

            loss = model.compute_loss(
                yolo_loss=yolo_out.loss,
                attributes_pred=attributes_pred,
                attributes_true=attr_labels,
                index_pred=index_pred,
                index_true=index_values,
                quad_pred=quad_pred,
                quad_true=quad_values
            )
            val_metrics["loss"] += loss.item()

            batch_metrics = calculate_metrics(attributes_pred, attr_labels, index_pred, index_values, quad_pred, quad_values)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]

    for key in val_metrics:
        val_metrics[key] /= num_batches

    print(f"Validation | Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1 Score: {val_metrics['f1_score']:.4f}")

# Save model weights
end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_path = f"logs/checkpoints/weights-{end_time}.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved at {checkpoint_path}")