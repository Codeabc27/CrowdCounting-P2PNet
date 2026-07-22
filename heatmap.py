import numpy as np
import cv2


def create_density_heatmap(points, width, height):

    # Empty density map
    density = np.zeros((height, width), dtype=np.float32)

    # Add points
    for point in points:
        x = int(point[0])

        y = int(point[1])

        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1

    # Gaussian smoothing
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=15, sigmaY=15)

    # Normalize
    if density.max() > 0:
        density_normalized = density / density.max() * 255

    else:
        density_normalized = density

    density_normalized = density_normalized.astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)

    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap, density
