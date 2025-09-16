import numpy as np

def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), n_points)

    # interpolate x and y coordinates
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
    ]
    
    return xy_interp

def iterative_resample_polygon(xy: np.ndarray, n_points: int = 100, n_iter: int = 3) -> np.ndarray:
    # resample multiple times to numerically stabilize the result to be truly equidistant
    xy_resampled = resample_polygon(xy, n_points)
    for _ in range(n_iter-1):
        xy_resampled = resample_polygon(xy_resampled, n_points)
    return xy_resampled


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    coords = [
        {'x': 354.0, 'y': 424.0}, {'x': 318.0, 'y': 455.0}, {'x': 299.0, 'y': 458.0}, {'x': 284.0, 'y': 464.0}, {'x': 250.0, 'y': 490.0},
        {'x': 229.0, 'y': 492.0}, {'x': 204.0, 'y': 484.0}, {'x': 187.0, 'y': 469.0}, {'x': 176.0, 'y': 449.0}, {'x': 164.0, 'y': 435.0},
        {'x': 119.0, 'y': 274.0}, {'x': 121.0, 'y': 264.0}, {'x': 118.0, 'y': 249.0}, {'x': 118.0, 'y': 224.0}, {'x': 121.0, 'y': 209.0},
        {'x': 130.0, 'y': 194.0}, {'x': 138.0, 'y': 159.0}, {'x': 147.0, 'y': 139.0}, {'x': 155.0, 'y': 112.0}, {'x': 170.0, 'y': 89.0},
        {'x': 190.0, 'y': 67.0}, {'x': 220.0, 'y': 54.0}, {'x': 280.0, 'y': 47.0}, {'x': 310.0, 'y': 55.0}, {'x': 330.0, 'y': 56.0},
        {'x': 345.0, 'y': 60.0}, {'x': 355.0, 'y': 67.0}, {'x': 367.0, 'y': 80.0}, {'x': 375.0, 'y': 84.0}, {'x': 382.0, 'y': 95.0},
    ]

    # construct numpy array from list of dicts
    xy = np.array([(c['x'], c['y']) for c in coords])

    n_points = 30
    # resample polygon
    print(f"Resampling polygon with {len(xy)} points to {n_points} points")
    start_time = time.time()
    xy_resampled = iterative_resample_polygon(xy, n_points, n_iter=20)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # plot result
    fig, ax = plt.subplots(figsize=(7,14))
    ax.scatter(xy[:, 1], xy[:, 0], marker='o', s=150, label='original', color='black')
    ax.scatter(xy_resampled[:, 1], xy_resampled[:, 0], label='resampled', color='red')
    ax.set_aspect(1)
    ax.invert_yaxis()
    plt.legend()
    plt.show()

    # Calculate distances between consecutive vertices
    distances = np.sqrt(np.sum((xy_resampled[1:] - xy_resampled[:-1])**2, axis=1))
    print('Distance between consecutive vertices:', distances)

    

