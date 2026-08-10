# stereo-vision

## epipolar geometry
1. find the matching points
2. compute essential matrix and fundamental matrix
3. compute the rotation and translation matrix
4. run 3d trianglulation

### epipolar search
1. pre-requisite: need to have fundamental matrix
2. find the seed point in left image
3. form the epiline in right image
4. search for the best matches along the epiline in the right image

### block matching
1. pre-requisite: need to rectify the images
2. 

## dataset
- https://www.eth3d.net/datasets
- http://lightfield.stanford.edu/lfs.html
- https://github.com/opencv/opencv/blob/4.x/samples/data/
- https://vision.middlebury.edu/stereo/data/scenes2021/

## reference
### epipolar geometry
- https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
- https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf