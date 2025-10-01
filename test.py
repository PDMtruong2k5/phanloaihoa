# ==============================
# Nhận diện hoa Oxford Flowers 102 với PyTorch
# Code gốc được viết lại, thêm chú thích tiếng Việt và in kết quả tiếng Việt
# ==============================

import os, warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ---------- 1. Cấu hình thiết bị ----------
# Kiểm tra có GPU CUDA không, nếu có thì dùng, nếu không thì dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang sử dụng thiết bị: {device}")

# ---------- 2. Đường dẫn dữ liệu ----------
# Thư mục dữ liệu (sửa nếu cần)
DATA_DIR = "/kaggle/input/oxford-flowers-102"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Kiểm tra cấu trúc thư mục
for path, name in [(TRAIN_DIR, "Train"), (VALID_DIR, "Valid"), (TEST_DIR, "Test")]:
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if not f.startswith('.')])
        print(f"✅ Thư mục {name}: có {count} mục")
    else:
        print(f"❌ Không tìm thấy thư mục {name}: {path}")

# ---------- 3. Tiền xử lý ảnh ----------
# Tập train: có tăng cường dữ liệu (Data Augmentation)
train_transform = transforms.Compose([
    transforms.Resize(256),                         # Đổi kích thước ảnh
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)), # Cắt ngẫu nhiên
    transforms.RandomHorizontalFlip(p=0.5),        # Lật ngang ngẫu nhiên
    transforms.RandomRotation(30),                 # Xoay ngẫu nhiên
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.15), # Biến đổi màu
    transforms.RandomGrayscale(p=0.05),            # Ngẫu nhiên chuyển sang ảnh xám
    transforms.ToTensor(),                         # Đưa về tensor [0,1]
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])  # Chuẩn hóa theo ImageNet
])

# Tập valid/test: chỉ resize và chuẩn hóa, không augment
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------- 4. Tạo Dataset & DataLoader ----------
# Dùng ImageFolder cho train/valid
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_ds = datasets.ImageFolder(VALID_DIR, transform=val_transform)

# batch_size lớn hơn khi có GPU
batch_size = 32 if device.type == 'cuda' else 16
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# Tạo Dataset custom cho test (chỉ có ảnh, không có nhãn)
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname

test_ds = TestDataset(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# Thông tin dữ liệu
num_classes = len(train_ds.classes)
print(f"📊 Train: {len(train_ds)} ảnh, Valid: {len(valid_ds)} ảnh, Test: {len(test_ds)} ảnh")
print(f"🎯 Số lớp phân loại: {num_classes}")

# Bản đồ ngược từ index -> class
inv_map = {v: int(k) for k,v in train_ds.class_to_idx.items()}
print(f"🔄 Ví dụ ánh xạ class: {dict(list(inv_map.items())[:5])}")

# ---------- 5. Hàm tạo mô hình ----------
def create_model(model_name, num_classes, use_pretrained=True):
    """Tạo model với nhiều kiến trúc khác nhau"""
    if model_name == 'resnet50':
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
        except:
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'efficientnet':
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        except:
            # fallback về resnet34 nếu không có EfficientNet
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:  # mặc định dùng resnet18
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if use_pretrained else None
            model = models.resnet18(weights=weights)
        except:
            model = models.resnet18(weights=None)
            use_pretrained = False

        # Thay fc bằng classifier mới
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    return model, use_pretrained

# Chọn model theo thiết bị
if device.type == 'cuda':
    model_name = 'resnet50'
else:
    model_name = 'resnet18'

model, has_pretrained = create_model(model_name, num_classes, use_pretrained=True)
model = model.to(device)
print(f"🔥 Sử dụng {model_name} {'(có pretrained)' if has_pretrained else '(train từ đầu)'}")

# ---------- 6. Thiết lập huấn luyện ----------
criterion = nn.CrossEntropyLoss()

if has_pretrained:
    # Đóng băng các layer sớm, chỉ fine-tune layer cuối
    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            if 'layer4' not in name and 'features.7' not in name:
                param.requires_grad = False

    optimizer = optim.Adam([
        {'params': [p for n,p in model.named_parameters() if 'fc' in n or 'classifier' in n], 'lr':1e-3},
        {'params': [p for n,p in model.named_parameters() if ('layer4' in n or 'features.7' in n) and 'fc' not in n], 'lr':1e-4}
    ], weight_decay=1e-4)
else:
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ---------- 7. Vòng lặp huấn luyện ----------
best_val_acc = 0.0
patience_counter = 0
max_patience = 7

EPOCHS = 25 if device.type == 'cuda' else 15
print(f"🚀 Bắt đầu huấn luyện tối đa {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    # Huấn luyện
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss/total
    train_acc = correct/total

    # Validation
    model.eval()
    v_total, v_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
    val_acc = v_correct/v_total

    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
        print(f"🎉 Epoch {epoch+1}: Cải thiện! Train acc={train_acc:.3f}, Val acc={val_acc:.3f}")
    else:
        patience_counter += 1
        print(f"📊 Epoch {epoch+1}: Train acc={train_acc:.3f}, Val acc={val_acc:.3f} (patience={patience_counter})")

    scheduler.step(val_acc)

    # Early stopping
    if patience_counter >= max_patience:
        print(f"⏹️ Dừng sớm tại epoch {epoch+1}")
        break

print(f"🏆 Độ chính xác cao nhất trên tập Validation: {best_val_acc:.4f}")

# ---------- 8. Dự đoán trên tập Test ----------
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("✅ Đã load model tốt nhất")

model.eval()
ids, classes_out = [], []
print("🔮 Đang dự đoán trên tập Test...")

with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        for p, fname in zip(preds, fnames):
            class_id = inv_map[int(p)]
            ids.append(fname)
            classes_out.append(int(class_id))

# ---------- 9. Tạo file nộp ----------
submission_df = pd.DataFrame({
    "id": ids,
    "class": classes_out
})

print(f"📋 Kích thước submission: {submission_df.shape}")
print(f"🔢 Khoảng class dự đoán: {submission_df['class'].min()} - {submission_df['class'].max()}")
print("📄 Xem vài dòng đầu của submission:")
print(submission_df.head())

submission_df.to_csv("submission.csv", index=False)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

print("💾 Đã lưu file submission.csv thành công!")
print(f"✅ Sẵn sàng nộp với {len(submission_df)} dự đoán")
