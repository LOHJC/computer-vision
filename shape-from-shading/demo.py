
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

IMG = "imgs/img1.jpg"

if __name__ == "__main__":
    
    # load image and convert to grayscale
    img = cv.imread(IMG)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # brightness equation setup
    # I(x,y) = rho * (n(x,y) dot s)
    rho = 1 # assume 1 for simplicity
    # I(x,y) = (n(x,y) dot s)
    s = np.array([0, 0, 1]) # light direct from camera
    # I approximate = nz
    
    # surface normal estimation (from image gradient)
    gx = cv.Sobel(img_gray, cv.CV_32F, 1, 0, ksize=3) # dz/dx
    gy = cv.Sobel(img_gray, cv.CV_32F, 0, 1, ksize=3) # dz/dy
    nx = -gx
    ny = -gy
    nz = np.ones_like(img_gray)

    print(nx.shape, ny.shape, nz.shape)
    normals = np.dstack((nx, ny, nz))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) # normalize

    normals_vis = ((normals + 1) / 2 * 255).astype(np.uint8)

    predicted_intensity = np.sum(normals * s, axis=2)
    error = img_gray.astype(np.float32)/255.0 - predicted_intensity
    print("Mean shading error:", np.mean(np.abs(error)))

    # depth calculation
    p = -nx / (nz + 1e-6) # gx
    q = -ny / (nz + 1e-6) # gy

    # simple cumsum
    depth = np.cumsum(p, axis=1) + np.cumsum(q, axis=0)
    depth_norm = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # visualize in point cloud    
    h, w = depth_norm.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_norm
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    colors = img_rgb.reshape(-1, 3) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

        
    

