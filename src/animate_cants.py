import matplotlib.pyplot as plt

def animate(self, ants_paths) -> None:
    """
    Plot CANTS search space
    """
    points = []
    for level, in_space in enumerate(self.space.inputs_space.inputs_space.values()):
        for pnt in in_space.points:
            points.append([pnt.pos_x, pnt.pos_y, pnt.pos_l, pnt.pheromone])
    for pnt in self.space.output_space.points:
        points.append([pnt.pos_x, pnt.pos_y, pnt.pos_l, pnt.pheromone])
    for pnt in self.space.all_points.values():
        points.append([pnt.pos_x, pnt.pos_y, pnt.pos_l, pnt.pheromone])


    points = np.array(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=points[:, 3] * 10,
        c=points[:, 3],
        cmap="copper",
    )
    for path in ants_paths:
        pnts = []
        for pnt in path[:-1]:
            pnts.append([pnt.pos_x, pnt.pos_y, pnt.pos_l])
        pnts.append([path[-1].pos_x, path[-1].pos_y, self.space.time_lags])
        pnts = np.array(pnts)
        plt.plot(pnts[:, 0], pnts[:, 1], pnts[:, 2])

    plt.show(block=False)
    plt.pause(0.001)
    plt.close()
    plt.cla()
    plt.clf()

