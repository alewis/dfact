


    # if jnp.iscomplex(x):
        # return __house_complex(x)
    # else:
        # if x.size == 1:
            # return house_size_1(x)

        # a = x[1:]
        # sigma = jnp.dot(a, a)
        # v = jnp.ones(x.shape, x.dtype)

        # x0 = x[0]
        # if sigma == 0: # and np.abs(x0) >= 0:
            # beta = 0.
        # else:
            # mu = jnp.sqrt(x0**2 + sigma)
            # if x[0] <= 0.:
                # v0 = x0 - mu
            # else:
                # v0 = -sigma / (x0 + mu)
            # beta = (2*(v0**2))/(sigma + v0**2)
        # v = index_update(v, index[1:], a/v0)
        # output = [v, beta]
        # return output

# def __house_complex(x):
    # # if x.size==1:
        # # return [np.array([1.+0.j]), 0]

    # x0 = x[0]
    # x2 = jnp.dot(dag(x), x)
    # r = (jnp.abs(x0))**2
    # theta = jnp.angle(x0)
    # ei_theta = jnp.abs(jnp.exp(1.0j*theta))
    # trm = x2 * ei_theta
    # v0 = x0 + trm
    # v02 = np.dot(dag(v0), v0)
    # beta = 2/v02
    # v = jnp.ones(x.shape, x.dtype)
    # v = index_update(v, index[1:], x[1:]/v0)
    # return v


# def house_size_1(x):
    # """
    # Specialization of 'house' to size-1 input.
    # """
    # if x[0] >= 0:
        # beta = 0.
    # elif x[0] < 0:
        # beta = -2.
    # beta = 0.
    # v = jnp.array([1.]).astype(x.dtype)
    # output = [v, beta]
    # return output
