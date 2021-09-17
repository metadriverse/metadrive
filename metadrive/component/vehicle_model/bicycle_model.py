def plant_model(self, dt, *control):
    # in this model, we formulate the car's dynamic model as Bicycle model
    #
    # information that should be tracked in 'state'
    # x, y: geometric coordinate
    # v : velocity
    # phi:  heading angle
    # beta: velocity angle

    # things need to be finetuned
    # mass
    # mu*g: a = f/m + \mu * mg
    #

    [x, y, v, phi, beta] = self.state
    pedal, steering = control[0]
    pedal *= 500
    steering *= 40
    new_beta = np.arctan(0.5 * np.tan(steering / 180 * np.pi))

    if pedal < 0:
        pedal *= 3

    a = pedal / 500 * 3
    af = .5

    new_v = 0
    if v > 1e-5 or a > af:
        new_v = v + (a - af) * dt
        if v * new_v < 0:
            new_v = 0

    new_phi = phi + v / 2 / 2 * np.tan(steering / 180 * np.pi) * dt
    new_x = x + v * np.cos(phi + beta) * dt
    new_y = y + v * np.sin(phi + beta) * dt
    new_state = [new_x, new_y, new_v, new_phi, new_beta]
    self.state = new_state
    # self.chassis_np2.setPos(50, 0, 1)
    print('---------------', pedal, steering, self.state)
    self.chassis_np2.setPos(self.state[0], self.state[1], 1)
    self.chassis_np2.setHpr(90 + phi / np.pi * 180, 0, 0)

    # print('p s %d %d %.4f %.4f,'% (pedal, steering, beta, phi))
    # print('MPC model %.4f %.4f'%(off/dt,numpy.linalg.norm(new_state[2])))
    # print('MPC acceleration %.4f ' %(a))
    return new_state