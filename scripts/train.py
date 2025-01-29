import torch
import torch.optim as optim
import os
import datetime
from data_loader import train_loader, val_loader
from yolo_wrapper import YOLOWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOWrapper("yolov8s.pt", num_classes=1).model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

os.makedirs("logs/checkpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    num_train_batches = len(train_loader)

    for images, bboxes, *_ in train_loader:
        images = images.to(device)
        bboxes = [b.to(device) for b in bboxes]

        optimizer.zero_grad()

        preds = model(images)
        loss = model.loss(preds, [{"boxes": b[:, 1:], "labels": b[:, 0].long()} for b in bboxes])

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / num_train_batches
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")

    model.eval()
    YOLOWrapper.validate(YOLOWrapper, val_loader, device)

end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_path = f"logs/checkpoints/weights-{end_time}.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved at {checkpoint_path}")
