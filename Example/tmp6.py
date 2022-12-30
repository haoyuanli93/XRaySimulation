import numpy as np

k0 = 0


# ------------------------------------
# Get the rotation matrix
# ------------------------------------
kVec_for_rotmat = np.outer(np.ones(5), k0)
kVec_for_rotmat[1, 1] += np.pi * 2 / (dy_new * ny_new)
kVec_for_rotmat[2, 1] -= np.pi * 2 / (dy_new * ny_new)
kVec_for_rotmat[3, 2] += np.pi * 2 / (dz_new * nz_new)
kVec_for_rotmat[4, 2] -= np.pi * 2 / (dz_new * nz_new)

# Get the output wave-vector from the devices
output = []
for kVec in range(5):
    tmp = DeviceSimu.get_kout_multi_device(device_list=(yCrystals[0].crystal_list
                                                        + myCrystals[1].crystal_list)
    kin = k0Vec)
    output.append(np.copy(tmp[-1]))
kVec_for_rotmat = np.vstack((x for x in output))

# Get the output wave-vector
u1 = (kVec_for_rotmat[2, :] - kVec_for_rotmat[1, :]) / 2.
u2 = (kVec_for_rotmat[4, :] - kVec_for_rotmat[3, :]) / 2.

u_mat = np.zeros((2, 2))
u_mat[0] = u1[1:]
u_mat[1] = u2[1:]
u_mat /= 2 * np.pi
u_mat_inv = np.linalg.inv(u_mat)

# --------------------------------------------
#  Interpolation
# --------------------------------------------
# Define the grid size after the interpolation
new_ny2 = ny_new
new_nz2 = nz_new

# Define the position grid before the interpolation
new_position_grid = np.zeros((new_ny2, new_nz2, 2))
new_position_grid[:, :, 0] = np.arange(-new_ny2 // 2, new_ny2 // 2)[:, np.newaxis] * new_dy2
new_position_grid[:, :, 1] = np.arange(-new_nz2 // 2, new_nz2 // 2)[np.newaxis, :] * new_dz2
new_position_grid = np.reshape(new_position_grid, (new_ny2 * new_nz2, 2))
new_position_grid = np.dot(u_mat, new_position_grid.T).T

# For each x interpolate the y-z plane
for xIdx in range(nx_new):

    # Load and assemble the eField before interpolation
    eFieldYZslice = []
    for yIdx in range(batchNumberY):
        with h5.File("./output/batchIdx_{}_yIdx_{}.h5".format(pulseIdx, yBatchIdx), 'rb') as h5file:
            eFieldYZslice.append(h5file['eFieldSlice'][xIdx, :, :])
    eFieldYZslice = np.concatenate(eFieldYZslice, axis=0)

    # Perform the interpolation
    field_fit_mag = interpolate.interpn(points=(np.arange(-new_ny2 // 2, new_ny2 // 2) / new_ny2,
                                                np.arange(-new_nz2 // 2, new_nz2 // 2) / new_nz2),
                                        values=np.abs(field_z_holder[idx]),
                                        xi=new_position_grid,
                                        method='linear',
                                        bounds_error=False,
                                        fill_value=0.)

    field_fit_phase = interpolate.interpn(points=(np.arange(-new_ny2 // 2, new_ny2 // 2) / new_ny2,
                                                  np.arange(-new_nz2 // 2, new_nz2 // 2) / new_nz2),
                                          values=unwrap_phase(np.angle(field_z_holder[idx])),
                                          xi=new_position_grid,
                                          method='linear',
                                          bounds_error=False,
                                          fill_value=0.)

    field_fit[idx] = np.reshape(field_fit_mag * np.exp(1.j * field_fit_phase), (new_ny2, new_nz2))

