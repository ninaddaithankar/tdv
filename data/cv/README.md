## Instructions for downloading each of the datasets

### ImageNet, RedPajama, MNIST

These three downloads are managed through [HuggingFace](https://huggingface.co/). You will need to create an account and provide credentials by adding your HuggingFace token to the environment before running any download. In particular, these two environment variables are required:
1. `HF_TOKEN`: Set this to the User Access Token in your HuggingFace profile
2. `HF_HOME`: Set this to the directory you would like HuggingFace downloads to be stored

See more information on environment setup at [HuggingFace's official documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

### Note on `ffprobe` for Computer Vision datasets

We use ffprobe to read the duration of each video in our cv dataloaders. While opencv-python does not require a library installation external to Python, we found opencv to be more unreliable than ffprobe at reading video durations.

#### ffprobe setup: 

If using conda, easiest way to install ffprobe is by running `conda install -c conda-forge libiconv ffmpeg` (libiconv is needed for ssv2 dataloader else the video metadata extraction will fail)

On a linux system, you can also `apt install ffmpeg` to get ffprobe. If ffprobe is in your path, the training scripts should run without any additional setup for ffprobe.

If that does not work, you can download ffprobe at the [ffmpeg download site](https://ffmpeg.org//download.html). Once downloaded, extract the binaries and provide the path to the ffprobe file by either setting:
- the environment variable `FFPROBE_PATH=<path_to_ffprobe>`
- or the command-line argument `--ffprobe_path=<path_to_ffprobe>`

### Kinetics-400 & Kinetics-600

To download Kinetics-400 and Kinetics-600, use the scripts at https://github.com/cvdfoundation/kinetics-dataset. Set the command-line argument `--dataset_dir=<path_to_dataset>`.

### Something-something-v2
SSv2 is distributed by Qualcomm in separate files at [this link](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads). Download the video files and the labels. After downloading all the video files, you can concatenate them and unzip them as a single file:

```
cat 20bn-something-something-v2-* > ssv2_archive
tar -xvf ssv2_archive
```
Also unzip the labels.
Then, set the command-line argument `--dataset_dir=<path_to_dataset>`.
### UCF-101

Download the UCF dataset and annotations from UCF's website: https://www.crcv.ucf.edu/data/UCF101.php. *Note: as www.crcv.ucf.edu does not have a trusted certificate, if using wget to download you must provide the argument `--no-check-certificate`.*

The UCF dataset directory structure should appear as:
- UCF101
    - ucfTrainTestlist/ (UCF dataset annotations)
    - *.avi (all UCF dataset videos)

Set the command-line argument `--dataset_dir=<path_to_dataset>`.

### Ego4D

The Ego4D dataset is hosted by Meta AI and requires access approval through the [Ego4D official website](https://ego4d-data.org/). Once your request is approved, you will receive secure download links and credentials for the videos and annotations.

#### Steps:
1. Visit <https://ego4d-data.org/> and request access using your institutional or organizational email.  
2. After approval, follow the instructions provided by the Ego4D team to download the dataset via `aws s3` or `wget`.  
3. The dataset includes multiple benchmarks (e.g., *Short-Term Anticipation*, *Long-Term Anticipation*, *Forecasting*, etc.). For TDV pretraining and evaluation, use the **Full-Scale Video** subset.  
4. Once downloaded, extract all archives into a single directory structure:
    ```
    ego4d/
        ├── videos/
        ├── annotations/
        ├── metadata/
    ```
5. If using preprocessed or compressed Ego4D features, set:
    ```
    --processed_dataset_dir=<path_to_preprocessed_dataset>
    ```
    Otherwise, for raw videos, set:
    ```
    --dataset_dir=<path_to_dataset>
    ```

#### Notes:
- Ego4D uses `ffprobe` for video duration parsing (see the ffprobe section above).  
- Ensure sufficient storage — the complete Ego4D dataset requires **over 2 TB** of space.  
- The dataset is released under a **non-commercial academic use license**.

