# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2  # This is OpenCV, our video-reading tool
import os
import random
from glob import glob # This is a handy tool for finding files

# --- 1. Configuration: Set our main variables ---

# We'll grab 16 frames from each video
NUM_FRAMES = 16
# We'll resize each frame to 112x112
FRAME_HEIGHT = 112
FRAME_WIDTH = 112

# --- 2. The "Assembly Line" (Our Dataset Class) ---
# This class teaches PyTorch how to get *one* item from our dataset.

class VideoDataset(Dataset):
    """
    This class loads video files, samples frames, applies transforms,
    and returns a tensor clip and its label.
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the action folders.
            split (string): 'train' or 'val' to decide which data to load.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Get all class names (folder names) automatically
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create a mapping from class name to a number (label)
        # e.g., {'BrushingTeeth': 0, 'HorseRiding': 1, ...}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # This will hold all our data samples (video path, label)
        self.samples = []
        
        # Now, we find all video files and split them
        self._make_dataset()

    def _make_dataset(self):
        """
        This is a helper function to find all videos and split them.
        We'll do a simple 80% train, 20% validation split for each class.
        """
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_path = os.path.join(self.root_dir, class_name)
            
            # Find all .avi files in the class folder
            video_files = sorted(glob(os.path.join(class_path, "*.avi")))
            
            # Shuffle them to make the split random
            random.shuffle(video_files)
            
            # Split point: 80% of the files
            split_idx = int(len(video_files) * 0.8)
            
            if self.split == 'train':
                # Get the first 80%
                self.samples.extend([(path, class_idx) for path in video_files[:split_idx]])
            else: # self.split == 'val'
                # Get the last 20%
                self.samples.extend([(path, class_idx) for path in video_files[split_idx:]])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This is the most important function.
        It gets one 'item' (a video clip) from our dataset.
        """
        # Get the path to the video and its label
        video_path, label = self.samples[idx]
        
        # --- This is the core video processing logic ---
        frames = self.load_video_frames(video_path)
        
        # Apply transformations (like resize, crop, normalize)
        if self.transform:
            # We need to apply the *same* transform to *all* frames in the clip
            # We'll use a little trick to do this
            # Note: We are assuming transforms expect a PIL Image.
            # We'll need to convert our OpenCV (numpy) frames to PIL.
            
            # from PIL import Image
            # pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
            # transformed_frames = [self.transform(frame) for frame in pil_frames]
            
            # A more direct way with tensors (let's build this into the transform)
            # For now, let's pass the list of frames to the transform
            frames = self.transform(frames)
            
        # The model expects [Channels, Num_Frames, Height, Width]
        # Our frames tensor is [Num_Frames, Channels, Height, Width]
        # We use .permute() to swap the dimensions
        frames = frames.permute(1, 0, 2, 3) 
        
        return frames, label

    def load_video_frames(self, video_path):
        """
        Loads, samples, and preprocesses frames from a video file.
        """
        frames = []
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(video_path)
        
        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: Could not read video {video_path}. Skipping.")
            return torch.zeros(NUM_FRAMES, 3, FRAME_HEIGHT, FRAME_WIDTH) # Return an empty tensor

        # Calculate indices of the frames to sample
        # We pick NUM_FRAMES frames, evenly spaced
        indices = torch.linspace(0, total_frames - 1, NUM_FRAMES).long()
        
        for i in indices:
            # Set the video capture to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
            ret, frame = cap.read()
            
            if not ret:
                # If frame can't be read, use the previous frame
                if len(frames) > 0:
                    frame = frames[-1] 
                else:
                    # If it's the very first frame, use a black image
                    frame = torch.zeros(FRAME_HEIGHT, FRAME_WIDTH, 3).numpy().astype('uint8')
            
            # --- Preprocessing the frame ---
            # 1. Resize the frame
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            # 2. Convert from BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
            
        cap.release()
        
        # Stack all frames into a single numpy array
        # Shape will be (NUM_FRAMES, H, W, Channels)
        frames_np = torch.tensor(frames, dtype=torch.uint8) # (16, 112, 112, 3)
        
        # We need to change it to (NUM_FRAMES, Channels, H, W)
        frames_np = frames_np.permute(0, 3, 1, 2) # (16, 3, 112, 112)
        
        return frames_np.float() / 255.0 # Convert to float and normalize to [0, 1]


# --- 3. Define our Transforms (Augmentation and Normalization) ---

# For a video, we need to apply the *same* random augmentation to *all* frames
# For example, if we horizontally flip one frame, we must flip all 16.

# We'll use pre-computed mean and std for video datasets (Kinetics)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

# Define the transforms
# We'll create a dictionary for 'train' and 'val' transforms
data_transforms = {
    'train': transforms.Compose([
        # We did resizing in the `load_video_frames` function.
        # Here we'll do normalization and augmentation.
        
        # Apply the same horizontal flip to all frames in the clip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Normalize each frame
        transforms.Normalize(KINETICS_MEAN, KINETICS_STD)
    ]),
    'val': transforms.Compose([
        # Validation data doesn't get random augmentations
        transforms.Normalize(KINETICS_MEAN, KINETICS_STD)
    ]),
}


# --- 4. Test it all out (The "Sanity Check") ---
if __name__ == '__main__':
    # This block only runs if you execute `python dataloader.py` directly
    
    # ***IMPORTANT: Change this path to your folder***
    DATA_DIR = "UCF101_new" # <--- CHANGE THIS to "ucf_7_classes"
    
    # 1. Create the training dataset
    train_dataset = VideoDataset(root_dir=DATA_DIR, split='train', transform=data_transforms['train'])
    
    # 2. Create the validation dataset
    val_dataset = VideoDataset(root_dir=DATA_DIR, split='val', transform=data_transforms['val'])

    # 3. Create the DataLoaders (the "forklifts")
    # Batch size is how many clips to load at once
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"--- Dataset Stats ---")
    print(f"Classes: {train_dataset.classes}")
    print(f"Num classes: {len(train_dataset.classes)}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")

    print(f"\n--- DataLoader Test ---")
    # Let's get one batch from the training loader
    try:
        video_batch, label_batch = next(iter(train_loader))
        
        print(f"Video batch shape: {video_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
        
        # The video batch shape should be:
        # [Batch_Size, Channels, Num_Frames, Height, Width]
        # e.g., [8, 3, 16, 112, 112]
        
        print(f"\nSuccessfully loaded one batch!")
        print(f"Labels in this batch: {label_batch.numpy()}")

    except Exception as e:
        print(f"\n--- ERROR loading data ---")
        print(f"An error occurred: {e}")
        print("Please check the following:")
        print(f"1. Is your DATA_DIR set correctly? (Currently: '{DATA_DIR}')")
        print(f"2. Does this folder contain sub-folders for each action?")
        print(f"3. Do those sub-folders contain .avi files?")
        print(f"4. Do you have 'opencv-python' installed? (`pip install opencv-python`)")