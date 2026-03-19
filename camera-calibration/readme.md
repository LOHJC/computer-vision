
# camera calibration

## objective
- get the intrinsic and extrinsic parameters of the camera to correctly interpret the visual information acquired

## steps
1. get a calibration pattern
    - this example chosen symmetric circles 
2. capture multiple images of it on the camera
3. detect the features point
    - in this example, each circle points
4. match the 2D image points to the 3D object points
    - in this example, assuming the pattern is lying flat (z=0), and camera are moving around
    - the x and y is by the coordinate
        - top left corner is (0,0)
        - top right corner is (X,0)
        - bottom left corner is (0,Y)
        - bottom right corner is (X,Y)
5. compute the camera parameters


## references
- https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html