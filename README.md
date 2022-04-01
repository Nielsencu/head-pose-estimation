# Attention Indexing Model

Real-time attention tracking using various metrics such as head pose estimation and blink and yawn detection using OpenCV and dlib. The head pose estimation model was originally forked from yinguobing's [head pose estimation repo](https://github.com/yinguobing/head-pose-estimation) which used Tensorflow to obtain the facial landmarks. However, we decided to use dlib's implementation instead for speed and less bloat. 

![demo](doc/demo.gif)
![demo](doc/demo1.gif)

## Getting Started

Git clone this repo into your local machine. 
```bash
git clone --depth=1 https://github.com/Nielsencu/ripplecreate-attention-model.git
```

For the Python version, `python==3.9` was used during testing but `python>=3.7` should work as well. 

Preferably, use Anaconda and install the packages into a conda environment. This is because `dlib` can be troublesome to install using a standard Python installation. 
```bash
pip install -r requirements.txt
```

## Running

A video file or a webcam index should be assigned through arguments. If no source isprovided, the built-in webcam will be used by default.

### With Video File

Works with any video format that OpenCV supports (`mp4`, `avi`, etc.):
```bash
python3 main.py --video /path/to/video.mp4
```

### With Webcam

The webcam index (to account for multiple webcams) should be provided:
```bash
python3 main.py --cam 0
```

## How It Works

1. Facial Landmark Detection. A non-deep-learning pre-trained predictor called `shape_predictor_68_face_landmarks.dat` takes in faces it find as inputs and outputs a set of 68 facial landmarks for each face found.
2. Pose Estimation. Using the 68 facial landmarks, the pose can be calculated by a mutual PnP algorithm. 
3. Blink Detection. Using the 68 facial landmarks, the blink rate and amount of time the eyes are closed can be calculated using a few of the landmarks. 
4. Yawn Detection. Using the 68 facial landmarks, a person's yawn can be calculated using a few of the landmarks. 
5. The scores collected will then be sent to a database for storage and dashboarding. 

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
The 3D face model comes from OpenFace, you can find the original file [here](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt).

The built-in face detector comes from OpenCV, you can find the repo [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector). 
