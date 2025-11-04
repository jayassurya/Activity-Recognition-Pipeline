# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# --- 1. Import our custom modules ---
from dataloader import VideoDataset, data_transforms
from model import create_model  

# --- 2. Configuration ---
# Your classes. Make sure this matches your folder name and dataloader.
NUM_CLASSES = 7
DATA_DIR = "UCF101_new " # <--- IMPORTANT: Make sure this is your data folder
SAVED_MODEL_PATH = "activity_model_best.pth"

# Training Hyperparameters
NUM_EPOCHS = 15      # How many full passes through the data
BATCH_SIZE = 4       # How many videos to process at once (lower this if you get "Out of Memory" errors)
LEARNING_RATE = 0.001 # How fast the model learns

def main():
    print("--- Starting Activity Recognition Training ---")

    # --- 3. Set up the Device (use GPU if available) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Create DataLoaders (The "Forklifts") ---
    print("Loading datasets...")
    train_dataset = VideoDataset(
        root_dir=DATA_DIR, 
        split='train', 
        transform=data_transforms['train']
    )
    
    val_dataset = VideoDataset(
        root_dir=DATA_DIR, 
        split='val', 
        transform=data_transforms['val']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True # Helps speed up data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- 5. Create the Model (The "Engine") ---
    model = create_model(num_classes=NUM_CLASSES)
    
    # Move the model to the GPU/CPU
    model = model.to(device)

    # --- 6. Define Loss Function and Optimizer (The "Teacher") ---
    
    # Loss Function: CrossEntropyLoss (as requested)
    # This is good for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam
    # We only want to train the *new head* (the `model.fc` layer)
    # We "froze" the other layers in model.py
    # We can tell the optimizer to only update parameters where requires_grad == True
    params_to_update = []
    print("\nParameters to be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t{name}")
            
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

    # --- 7. The Training Loop ---
    print(f"\n--- Starting Training for {NUM_EPOCHS} epochs ---")
    
    best_val_accuracy = 0.0 # Keep track of the best model

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train() # Set model to "training mode"
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            # Move data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: get predictions
            outputs = model(inputs)
            
            # 3. Calculate the loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass: calculate gradients
            loss.backward()
            
            # 5. Optimizer step: update the weights
            optimizer.step()
            
            # --- Statistics ---
            _, preds = torch.max(outputs, 1) # Get the class with the highest score
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval() # Set model to "evaluation mode"
        val_loss = 0.0
        val_corrects = 0
        
        # We don't need to calculate gradients during validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)

        # --- Print results for the epoch ---
        end_time = time.time()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {(end_time-start_time):.2f}s")
        print(f"\tTrain Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"\t Val Loss: {epoch_val_loss:.4f} |  Val Acc: {epoch_val_acc:.4f}")

        # --- Save the best model ---
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), SAVED_MODEL_PATH)
            print(f"\t *** New best model saved to {SAVED_MODEL_PATH} (Acc: {best_val_accuracy:.4f}) ***")

    print("\n--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

# --- 8. Run the main function ---
if __name__ == '__main__':
    main()