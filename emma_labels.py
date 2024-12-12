import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from emma_net.EMMA import EMMA

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, missing_columns):
        self.data = data
        self.missing_columns = missing_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = f"../ABAW_7th/cropped_aligned/{sample['image']}"
        image = self.load_image(image_path)  # Load and preprocess image

        # Convert inputs to float and handle invalid values
        inputs = np.array(image, dtype=np.float32)  # Ensure inputs are float32
        targets = sample[self.missing_columns].to_numpy(dtype=np.float32)  # Ensure targets are float32

        # Replace NaN in targets with 0 or another default value
        targets = np.nan_to_num(targets, nan=0.0)

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    def load_image(self, path):
        from PIL import Image
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match ViT input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize as per ViT expectations
        ])
        image = Image.open(path).convert("RGB")
        return preprocess(image)

# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device type: {device}")
    
    # Step 1: Load Data
    file_path = './emma_annotations/emma_training_set_annotations.csv'
    # columns = ['image', 'valence', 'arousal', 'expression'] + [f'AU{i}' for i in [1,2,4,6,7,10,12,15,23,24,25,26]]
    dtypes = {'image': str,
              'valence': float,
              'arousal': float,
              'expression': int,
              'au1': int,
              'au2': int,
              'au4': int,
              'au6': int,
              'au7': int,
              'au10': int,
              'au12': int,
              'au15': int,
              'au23': int,
              'au24': int,
              'au25': int,
              'au26': int }
    data = pd.read_csv(file_path, sep=',', decimal = '.', dtype=dtypes)

    # Step 2: Handle Missing Values
    missing_columns = ['valence', 'arousal', 'expression'] + [f'AU{i}' for i in [1,2,4,6,7,10,12,15,23,24,25,26]]
    data[missing_columns] = data[missing_columns].replace('10', np.nan).replace(10, np.nan).replace(10.0, np.nan)  # '10' represents missing values
    data[missing_columns] = data[missing_columns].astype(float)

    # Step 3: Prepare Dataset and DataLoader
    dataset = CustomDataset(data, missing_columns)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    # Step 4: Initialize and Train Model
    model = EMMA(num_classes=(1+12), exp_model_path = './checkpoints_ver2.0/affecnet8_epoch5_acc0.6209.pth').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
    criterion = torch.nn.MSELoss()

    # Enable Mixed Precision (Optional)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting training")

    # Training Loop
    for epoch in range(10):  # Replace with the desired number of epochs
        print(f"Epoch {epoch} started")
        model.train()
        for i, (inputs, targets) in enumerate(data_loader):
            if i % 10 == 0:  # Log every 10 batches
                print(f"Processing batch {i}")

            # Move inputs and targets to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            # Mixed Precision Training
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        print(f"Completed epoch {epoch}")
            
    print("Completed training!!")
    
    print("Starting prediction")

    # Step 5: Predict Missing Labels
    model.eval()
    with torch.no_grad():
        for idx, row in data[data[missing_columns].isnull().any(axis=1)].iterrows():
            image_path = f"../ABAW_7th/cropped_aligned/{row['image']}"
            image = dataset.load_image(image_path)
            inputs = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
            predictions = model(inputs)
            data.loc[idx, missing_columns] = predictions.cpu().numpy()  # Move predictions back to CPU


    print("Completed prediction!!")

    # Save Completed Data
    data.to_csv('completed_data.csv', index=False)