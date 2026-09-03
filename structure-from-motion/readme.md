# stucture from motion (sfm)

## references
- https://www.youtube.com/watch?v=iJTqlb7gsWY&t
- https://github.com/openMVG/SfM_quality_evaluation/tree/master
- https://github.com/muneebaadil/structure-from-motion/blob/master/README.md

## datasets
- ~~https://www.eth3d.net/datasets~~
- https://demuc.de/colmap/datasets/
  - south building (smallest set provided)
  - camera params are provided in `cameras.txt` and `images.txt` in `south-building/sparse`
  - 
- https://github.com/rshilliday/sfm/tree/master/datasets/Viking
  - image set using: Viking 
  - requires to get the camera param in the code `https://github.com/rshilliday/sfm/blob/master/matching.py`
    - `K = np.matrix('523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00')`
  - can download the dataset via `https://download-directory.github.io/`