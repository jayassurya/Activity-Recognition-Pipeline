# model.py
import torch
import torch.nn as nn
from torchvision.models import video

# --- 1. Define our Model Building Function ---

def create_model(num_classes):
    """
    Creates and returns a pre-trained R(2+1)D video model,
    fine-tuned for our number of classes.

    Args:
        num_classes (int): The number of classes for the final layer
                           (e.g., 7 for our dataset).
    """
    
    print("Loading pre-trained R(2+1)D_18 model...")
    # Load the pre-trained model
    # 'pretrained=True' downloads the weights trained on the Kinetics dataset
    model = video.r2plus1d_18(pretrained=True)
    
    # --- 2. Freeze the "Engine" ---
    # We "freeze" the parameters of the pre-trained layers.
    # This means we tell PyTorch: "Don't change these existing parts, 
    # they are already very smart."
    for param in model.parameters():
        param.requires_grad = False
        
    # --- 3. Replace the "Head" ---
    # The "head" is the final layer that makes the classification.
    # We need to replace it with a new one for our 7 classes.
    
    # Get the number of input features for the original head
    # In r2plus1d_18, this is model.fc
    num_ftrs = model.fc.in_features
    
    # Create our new, untrained final layer
    # This is the *only* part of the model that will be trained
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"Model head replaced. New head will output {num_classes} classes.")
    
    return model

# --- 4. Test it all out (The "Sanity Check") ---
if __name__ == '__main__':
    # This block only runs if you execute `python model.py` directly

    # --- FIX: Import variables from our dataloader FIRST ---
    # We need these variables to define the shape of our fake data
    from dataloader import NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH
    
    # Our number of classes
    NUM_CLASSES = 7 
    
    # 1. Create the model
    my_model = create_model(num_classes=NUM_CLASSES)
    
    # 2. Let's see the new head
    print("\n--- Model Head Structure ---")
    print(my_model.fc)

    # 3. Test with some "fake data"
    # This ensures the model can actually process our data shape.
    print(f"\n--- Model Forward Pass Test ---")
    
    # Create a fake "batch" of video clips
    # Shape: [Batch_Size, Channels, Num_Frames, Height, Width]
    fake_batch_size = 8
    fake_video = torch.randn(fake_batch_size, 3, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH) 
    
    print(f"Feeding in fake data with shape: {fake_video.shape}")
    
    try:
        # "Show" the fake video to the model
        output = my_model(fake_video)
        
        print(f"Successfully got output!")
        print(f"Output shape: {output.shape}")
        
        # The output shape should be [Batch_Size, Num_Classes]
        # e.g., [8, 7]
        if output.shape == (fake_batch_size, NUM_CLASSES):
            print("Test PASSED! Shape is correct.")
        else:
            print(f"Test FAILED! Expected shape [{fake_batch_size}, {NUM_CLASSES}] but got {output.shape}")

    except Exception as e:
        print(f"\n--- ERROR during model test ---")
        print(f"An error occurred: {e}")