# shape from shading

## objective 
3D reconstruction from a single 2D image by analyzing the light and shadows on the surface.

## steps
1. capture an image
2. define the condition
    - light source direction
    - surface albedo (reflectivity)
3. brightness equation setup
    - I(x,y) = rho * (n(x,y) dot s)
    - rho = surface albedo (reflectivity)
    - n(x,y) = surface normal
    - s = light source direction
4. surface normal estimation
5. depth integration
6. 3d shape output
