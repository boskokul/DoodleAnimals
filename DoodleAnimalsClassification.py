# Ovo ce biti notebook, izvrsavamo ga u google colab-u na gpu tesla t4 sa cuda, na google drive smo stavili 20animals.zip u folder dataset
import torch
from google.colab import drive
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# UCITAVANJE DATASET-A (ovo moramo ovako zbog google cloab-a da bismo pristupili samo tim nasim izabranim podacima, alternativa je keras api ali bi onda i duze trajalo i skidao bi sve zivotinje a ne samo ovih 20)
drive.mount('/content/drive')
# Pristupamo nasem google disku i na putanji dataset nalazimo zip-ovan ds koji unzip-ujemo u folder 20animals
!rm -rf 20animals
!unzip -q /content/drive/MyDrive/dataset/20animals.zip -d 20animals

class Config:
    DATA_DIR = '20animals'

    CLASSES = [
        'cat', 'cow', 'crocodile', 'dog', 'duck',
        'elephant', 'fish', 'hedgehog', 'horse', 'kangaroo',
        'lion', 'monkey', 'owl', 'panda', 'pig',
        'sheep', 'snail', 'snake', 'spider', 'zebra'
    ]

    TRAIN_SAMPLES_PER_CLASS = 2100
    VAL_SAMPLES_PER_CLASS = 450
    TEST_SAMPLES_PER_CLASS = 450

    # Hiperparametri
    BATCH_SIZE = 64
    NUM_EPOCHS = 30 # Povecano da bi se dalo prostora za Early Stopping
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    EARLY_STOPPING_PATIENCE = 5
    MIN_DELTA = 0.5 

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LR_SCHEDULER_PATIENCE = 3
    DROPOUT_RATE = 0.3

    MODEL_SAVE_PATH = 'densenet_doodle_best.pth'

config = Config()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, model_save_path, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model_save_path, model)
        elif (score < self.best_score + self.min_delta):
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f'Validation metric improved ({self.best_score:.4f} --> {score:.4f}).')
            self.best_score = score
            self.save_checkpoint(val_metric, model_save_path, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_metric, model_save_path, model):
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved! (Best Val Metric: {val_metric:.2f}%)')


# Da bismo koritili dataset za treniranje modela u pytorch-u moramo da napravimo dataloader-e a za to moramo prvo da nasledimo apstraktnu klasu Dataset i implementiramo njene metode __len__ i __getitem__
# parametar skip je samo indeks od kog ce krenuti da uzima slike za taj dataset a samples_per_class koliko
# parametar transform su augmentacije
class DoodleDataset(Dataset):
    def __init__(self, data_dir, classes, samples_per_class, transform=None, skip=0):
        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        self.labels = []

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"{class_dir} not found!")
                continue

            images = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
            selected_images = images[skip:skip + samples_per_class]

            for img_name in selected_images:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append(img_path)
                self.labels.append(class_idx)

        print(f"Loaded {len(self.samples)} pics from {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label
    
def create_dataloaders(config):
    # sto se tice transformcija ove Resize, Grayscale, ToTensor, Normalize su nam neophodne jer su modeli pretrenirani na imagenet skupu i imaju te velicine 224x224 i 3 boje a nas skup je crno beli crtezi
    # iako smo napisali u prijavi da verovatno nece trebati augmentacija dodao sam RandomRotation, RandomAffine, RandomHorizontalFlip jer su znacajno bas popravile rezultate (naravno samo u trening)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DoodleDataset(
        config.DATA_DIR, config.CLASSES, config.TRAIN_SAMPLES_PER_CLASS,
        transform=train_transform, skip=0
    )

    val_dataset = DoodleDataset(
        config.DATA_DIR, config.CLASSES, config.VAL_SAMPLES_PER_CLASS,
        transform=test_transform, skip=config.TRAIN_SAMPLES_PER_CLASS
    )

    test_dataset = DoodleDataset(
        config.DATA_DIR, config.CLASSES, config.TEST_SAMPLES_PER_CLASS,
        transform=test_transform,
        skip=config.TRAIN_SAMPLES_PER_CLASS + config.VAL_SAMPLES_PER_CLASS
    )

    # DataLoader je ono sto pytorch koristi za predaju podataka mrezi na treniranje, shuffle-ujemo za train podatke da ne nauci redosled
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_dataloaders(config)


# MODELI
def create_densenet_model(num_classes=20):
    # densenet121: 7M params

    model = models.densenet121(pretrained=True)

    # DenseNet ima 4 dense bloka - odmrznucemo poslednja 2

    # Prvo zamrzavanje svih
    for param in model.parameters():
        param.requires_grad = False

    # Odmrznemo denseblock3 i denseblock4 (poslednja dva bloka) i sloj za normalizaciju
    for name, module in model.features.named_children():
        if 'denseblock3' in name or 'denseblock4' in name or 'norm5' in name:
            for param in module.parameters():
                param.requires_grad = True

    # Zamenujemo classifier (FC layer)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(num_features, num_classes)
    )

    return model

model = create_densenet_model(
    num_classes=len(config.CLASSES)
)
model = model.to(config.DEVICE)

# Prikaz broja parametara
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")


# funkcija treniranja
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return running_loss / total, 100 * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total


# samo treniranje, za sad neka bude adam (videcemo jos) i ovi parametri za learning rate scheduler neka budu tako postavljeni pa cemo videti probleme sa overfittingom da ispravimo
# potrebno je doraditi da se implementira early stopping

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
# Scheduler prati loss, pa je mode='min'
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.LR_SCHEDULER_PATIENCE) 

early_stopper = EarlyStopper(
    patience=config.EARLY_STOPPING_PATIENCE,
    min_delta=config.MIN_DELTA,
)

train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(config.NUM_EPOCHS):
    print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}')

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # Learning rate scheduling
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss) # Scheduler prati loss
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")

    # provera ranog zaustavljanja
    if early_stopper(val_acc, config.MODEL_SAVE_PATH, model):
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

# Ovo je najveci val accuracy sto je postignut
best_val_acc = early_stopper.best_score 
print(f"Best validation accuracy: {best_val_acc:.2f}%")



# Evaluacione metrike i vizualizacija

model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(config.DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print('TEST SET REZULTATI')
print(f'Accuracy:  {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall:    {recall*100:.2f}%')
print(f'F1 Score:  {f1*100:.2f}%')

print('\nPo klasama:')
print(classification_report(all_labels, all_preds, target_names=config.CLASSES))

cm = confusion_matrix(all_labels, all_preds)

# Accuracy po klasama
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print('\nAccuracy po klasama:')
for i, class_name in enumerate(config.CLASSES):
    print(f'{class_name:12s}: {per_class_acc[i]*100:5.2f}%')


# Train i val loss i accuracy kroz epohe prikaz
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Ogranicavamo podatke za plotovanje na stvaran broj epoha (u slucaju ranog zaustavljanja)
epochs_ran = len(train_losses)
epochs_range = range(1, epochs_ran + 1)

ax1.plot(epochs_range, train_losses, label='Train Loss', linewidth=2)
ax1.plot(epochs_range, val_losses, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('DenseNet - Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, label='Train Acc', linewidth=2)
ax2.plot(epochs_range, val_accs, label='Val Acc', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('DenseNet - Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('densenet_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.CLASSES, yticklabels=config.CLASSES,
            cbar_kws={'label': 'Count'})
plt.title('DenseNet - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('densenet_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Accuracy po klasama bar chart
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(config.CLASSES)), per_class_acc * 100, color='steelblue', alpha=0.8)
ax.set_xlabel('Klasa', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('DenseNet - Accuracy po klasama', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(config.CLASSES)))
ax.set_xticklabels(config.CLASSES, rotation=45, ha='right')
ax.axhline(y=accuracy*100, color='r', linestyle='--', label=f'Overall: {accuracy*100:.2f}%')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('densenet_per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()