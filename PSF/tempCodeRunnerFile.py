 image = np.zeros((2 * h, 2 * h))
    for i in range(-h, h):
        for j in range(-h, h):
            if np.linalg.norm([i, j]) > radius:
                image[i + h, j + h] = 0
            else:
                image[i + h, j + h] = 1
    np.set_printoptions(threshold=np.inf,linewidth=200)
    print(h)
    print("image =",image)
    return image