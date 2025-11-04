# Activity-Recognition-Pipeline

Hi! This is my project for building an activity recognition system. I used PyTorch and a 3D CNN to teach a model how to tell the difference between 7 different actions from the UCF101 dataset.

You can feed it a new video, and it'll spit out a time-stamped log of what's happening.

## Features:

* **Pre-trained 3D Model:** I didn't train this from scratch (that would be crazy). I fine-tuned a pre-trained **R(2+1)D-18** model.
* **Time-Stamped Output:** The demo script (`demo.py`) processes a whole video and prints out what it thinks is happening and when.
* **Fixing the "Jitter":** The model would sometimes freak out for a split second. I added **temporal smoothing** (a "voting" system) to make the predictions stable.
* **Live Bar Chart:**  The demo window shows a live bar chart of what the model is "thinking" – you can see the probabilities for all 7 classes in real-time.
* **Final Summary:** When the video is done, it prints a final summary of all the action segments and pops up a `matplotlib` bar chart showing the percentage of the video for each action.



## Files Set Up

    .
    ├── UCF101_new/       # This is where I put my 7 folders of videos
    ├── dataloader.py     # This script preps all the video data
    ├── model.py          # This script defines the 3D model
    ├── train.py          # This is the script to actually train the model
    ├── demo.py           # This is the one you run to test a new video
    ├── activity_model_best.pth  # This is the final "brain" after training
    └── README.md         

## The Setup

1.  **Get the files** (clone this repo).
2.  **Make a virtual environment.** (You know, `python -m venv venv` and all that.)
3.  **Install these:**
    ```bash
    pip install torch torchvision
    pip install opencv-python numpy
    pip install matplotlib
    ```
4.  **Get the Data:**
    * I used a 7-class subset of the **UCF101 Dataset(Link= " https://www.crcv.ucf.edu/data/UCF101.php "**.
    * You'll have to download it, make your own `ucf_7_classes/` folder, and then copy the video folders you want to train on into it (like `Archery/`, `PullUps/`, etc.).

## How to Run It

### 1. Train the Model (The long part)

* Run `train.py`. This will load all your videos, train the model, and save the best version as `activity_model_best.pth`.
    ```bash
    python train.py
    ```
* **Heads up:** I have a 6GB 3060, so I had to set my `BATCH_SIZE` in `train.py` to `4`. If you get an "Out of Memory" error, set yours to `4` or `2`.

### 2. Run the Demo! (The fun part)

* Once you have your `.pth` file, open `demo.py`.
* Change the `INPUT_VIDEO_PATH` variable to whatever video you want to test.
* **SUPER IMPORTANT:** Make sure the `CLASS_NAMES` list in `demo.py` *exactly* matches your 7 folders, in alphabetical order.
    ```bash
    python demo.py
    ```
* A window will pop up, play your video with all the live predictions, and then give you the final summary when it's done.

## Results

I got about **[Your Accuracy, e.g., 88.5%]** accuracy on my validation set, which I'm pretty happy with! The demo works really well, especially after I fixed the jitter and the "unknown video" problem (more on that below).
