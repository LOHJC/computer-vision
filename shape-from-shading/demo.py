
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.fftpack import dst, idst

IMG = "imgs/img1.jpg"

def poisson_solver(p, q):
    """
    Solve Poisson equation for depth given slopes p=dz/dx, q=dz/dy
    using Discrete Sine Transform (DST).
    """
    h, w = p.shape

    # Compute divergence of gradient field
    f = np.zeros((h, w), dtype=np.float32)
    f[:, :-1] += p[:, :-1] - p[:, 1:]
    f[:-1, :] += q[:-1, :] - q[1:, :]

    # DST in both directions
    f_dst = dst(dst(f, type=1, axis=0), type=1, axis=1)

    # Eigenvalues for Poisson equation
    yy, xx = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')
    denom = (2*np.cos(np.pi*xx/(w+1)) - 2) + (2*np.cos(np.pi*yy/(h+1)) - 2)

    # Solve in frequency domain
    z_dst = f_dst / denom

    # Inverse DST
    z = idst(idst(z_dst, type=1, axis=0), type=1, axis=1)
    z /= (2*(h+1))*(2*(w+1))  # normalization

    return z


if __name__ == "__main__":
    # load image and convert to grayscale
    img = cv.imread(IMG)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5,5), 0)

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
    nx_n = normals[:,:,0]
    ny_n = normals[:,:,1]
    nz_n = normals[:,:,2]
    p = -nx_n / (nz_n + 1e-6)
    q = -ny_n / (nz_n + 1e-6)

    depth_use_cumsum = False
    if (depth_use_cumsum):
        # simple cumsum
        depth = np.cumsum(p, axis=1) + np.cumsum(q, axis=0)
    else:
        # poisson solver
        depth = poisson_solver(p, q)
        depth = -depth
    depth_norm = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # visualize in point cloud
    visualize_norm = True    
    if (visualize_norm):
        h, w = depth_norm.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth_norm
    else:
        h, w = depth.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth

    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    colors = img_rgb.reshape(-1, 3) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

        
    

