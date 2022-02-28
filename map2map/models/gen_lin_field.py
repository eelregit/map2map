import torch
from scipy.interpolate import interp1d
from ..data.norms.cosmology import D

def gen_lin_field(k, p, num_mesh_1d, box_length, z = 0.0, seed = None, sphere_mode = True) :
    '''
        Returns a random Gaussian realization of a linear vector field,
        sampled from the power spectrum interpolation table (k, p).

        k: array of wave numbers for interpolate, in units of [box_length] ** -1
        p: array of power spectrum values at k, in units of [k] ** -3
        num_mesh_1d: size of grid mesh along one dimension
        box_length: physical size of box (must be consistent with units of the power spectrum)
        z: redshift of power spectrum
        seed: random seed to use, if None a random one is drawn
        sphere_mode: if True sets all modes with k > nyquist to zero
    '''

    k = torch.tensor(k)
    p = torch.tensor(p)

    fundamental_mode = 2. * torch.pi / box_length
    num_modes_last_d = torch.floor(torch.tensor([num_mesh_1d]) / 2).to(int).item() + 1
    bin_volume = (box_length / num_mesh_1d) ** 3
    dis_std = 6
    lin_field_norm = dis_std * D(z) * bin_volume

    if seed is None :
        seed = torch.randint(2 ** 32 - 1, (1,)).item()

    random_generator = torch.Generator()
    random_generator.manual_seed(seed)

    scalar_potential = torch.fft.rfftn(torch.randn((num_mesh_1d, num_mesh_1d, num_mesh_1d), generator = random_generator)) / num_mesh_1d ** 1.5

    wave_numbers = torch.fft.fftfreq(num_mesh_1d, d = 1. / fundamental_mode / num_mesh_1d)
    k2_grid = (wave_numbers ** 2 + wave_numbers[:, None] ** 2 + wave_numbers[:, None, None] ** 2)[:, :, :num_modes_last_d]

    if sphere_mode :
        nyqiust_mode_sq = (fundamental_mode * (num_modes_last_d - 1)) ** 2
        zeros = k2_grid >= nyquist_mode_sq
        scalar_potential[zeros] = 0.
        k2_grid[zeros] = 0.
        del(zeros)
    nonzeros = k2_grid != 0.

    sigma = torch.sqrt(p * box_length ** 3)
    scalar_potential[nonzeros] *= torch.exp(torch.tensor(interp1d(torch.log(k), torch.log(sigma))(0.5 * torch.log(k2_grid[nonzeros]))))
    del(nonzeros)

    k2_grid[0,0,0] = 1
    scalar_potential /= -1j * k2_grid * lin_field_norm
    del(k2_grid)

    lin_field = torch.zeros((3, num_mesh_1d, num_mesh_1d, num_mesh_1d))
    lin_field[0,:,:,:] = torch.fft.irfftn(torch.reshape(wave_numbers, (num_mesh_1d, 1, 1)) * scalar_potential)
    lin_field[1,:,:,:] = torch.fft.irfftn(torch.reshape(wave_numbers, (1, num_mesh_1d, 1)) * scalar_potential)
    lin_field[2,:,:,:] = torch.fft.irfftn(torch.reshape(wave_numbers[:num_modes_last_d], (1, 1, num_modes_last_d)) * scalar_potential)

    return lin_field, seed
