# stucture from motion (sfm)
1. find matching points
2. find init 3d points
  - same step as [stereo vision](../stereo-vision/readme.md)
3. use perspective and point (pnp) to find the matching points in next image
  - basically matching the new 2d into the existing 3d
4. then triangulate again
6. [optimization] bundle adjustment 
5. keep repeating until everthing is done

## references
- https://www.youtube.com/watch?v=iJTqlb7gsWY&t
- https://github.com/openMVG/SfM_quality_evaluation/tree/master
- https://github.com/muneebaadil/structure-from-motion/blob/master/README.md
- https://colmap.github.io/index.html

## datasets
- ~~https://www.eth3d.net/datasets~~
- https://demuc.de/colmap/datasets/
  - image set using: south building (smallest set provided)
  - camera params are provided in `cameras.txt` and `images.txt` in `south-building/sparse`
- https://github.com/rshilliday/sfm/tree/master/datasets/Viking
  - image set using: viking
  - requires to get the camera param in the code `https://github.com/rshilliday/sfm/blob/master/matching.py`
    - `K = np.matrix('523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00')`
  - can download the dataset via `https://download-directory.github.io/`