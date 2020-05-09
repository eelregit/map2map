import numpy as np
import torch
import matplotlib.pyplot as plt

from nbodykit.lab import *
from nbodykit import setup_logging, style
from bigfile import BigFile
from scipy.special import hyp2f1


device = torch.device('cuda')

def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return a * hyp2f1(1, 1/3, 11/6, - OL * a**3 / Om) \
             / hyp2f1(1, 1/3, 11/6, - OL / Om)

z = 2  # FIXME
dis_norm = 6000 * D(z)  # [Mpc/h]


def dis2pos(dis_field,boxsize,Ng):
    """ Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape() = (Ng,Ng,Ng,3)
    """
    
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize
    
    pos = dis_field.copy()
    
    pos[...,2] += lattice
    pos[...,1] += lattice.reshape(-1, 1)
    pos[...,0] += lattice.reshape(-1, 1, 1) 
    
    pos[pos<0] += boxsize
    pos[pos>boxsize] -= boxsize
    
    return pos


def narrow_like(sr_box,tgt_Ng):
    """ sr_box in shape {3,Ng,Ng,Ng}, sr_box_Ng better to be even 
    """
    width = np.shape(sr_box)[1] - tgt_Ng
    
    assert width >= 0
    
    half_width = width // 2
    begin = half_width
    stop = tgt_Ng + half_width
    sr_box_cut = sr_box[:,begin:stop,begin:stop,begin:stop]

    return sr_box_cut


def cropfield(field,idx,reps,crop,pad): 
    """
    field in shape [Nc,Ng,Ng,Ng]  
    Need reps and crop prepared
    """        
    
    start = np.unravel_index(idx, reps) * crop  # find coordinate/index of idx on reps grid
    x = field.copy()    
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):            
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode='wrap')
    return x


def sr_field(model,lr_field,tgt_size): 
    """
    input unormalized lr_field in shape [Nc,Ng,Ng,Ng]
    return sr_field croped to tgt_size
    """
    lr_field = lr_field/dis_norm
    lr_field = np.expand_dims(lr_field, axis=0)
    lr_field = torch.from_numpy(lr_field).to(torch.float32)
    lr_field = lr_field.to(device)
    
    with torch.no_grad():
        sr_box = model(lr_field)   

    sr_box = sr_box.cpu().numpy()
    sr_box = sr_box[0]
    sr_box = sr_box*dis_norm
    
    sr_box = narrow_like(sr_box,tgt_size) 
    return sr_box


def lr2sr_Ps(lr_disp_path, tgt_ps_path, lr_ps_path, model, up_factor, pad, Lbox, n_split = 4):
    """
    input full lr displacement field in shape [Nc,Ng,Ng,Ng], pad is number, Lbox in Mpc
    chunk to n_split**3 pieces, upsample to sr field and then piece together
    get sr position field in shape [Ng**3, 3] and calculate Ps
    """
    lr_box = np.load(lr_disp_path)
    Ng_lr = np.shape(lr_box)[1]
    Ng_sr = Ng_lr*up_factor
    
    size = np.asarray(lr_box.shape[1:]) 
    
    ndim = len(size)
    pad = np.broadcast_to(pad, (ndim, 2)) 
    
    crop = np.broadcast_to(size // n_split, size.shape) 
    tgt_size = crop[0]*up_factor
    tgt_chunk = np.broadcast_to(tgt_size, size.shape)
    
    new_field = np.zeros([ndim,Ng_sr,Ng_sr,Ng_sr])  
    
    reps = size // crop   
    n_reps = int(np.prod(reps)) 
    
    #-----------------------------------------------
    # get sr position field
    
    for idx in range(0,n_reps):    
        chunk = cropfield(lr_box,idx,reps,crop,pad)
        chunk = sr_field(model,chunk,tgt_size)
        ns = np.unravel_index(idx, reps) * tgt_chunk  # new start point
        new_field[:,ns[0]:ns[0]+tgt_size,ns[1]:ns[1]+tgt_size,ns[2]:ns[2]+tgt_size] = chunk
        
    new_field = np.float32(new_field) # in shape (Nc,Ng,Ng,Ng)
    
    sr_pos = np.moveaxis(new_field, 0, -1) # to shape (Ng,Ng,Ng,3)
    sr_pos = dis2pos(sr_pos,Lbox*1000,Ng_sr) # to shape (Ng,Ng,Ng,3), Lbox convert to kpc
    sr_pos = np.moveaxis((np.moveaxis(sr_pos,-1,0)).reshape(3,-1),0,-1) # to shape (Ng**3,3)   
    
    #-----------------------------------------------
    # plot power spectrum
    
    f = ArrayCatalog({'Position': sr_pos*0.001})
    
    mesh = f.to_mesh(resampler='cic',Nmesh=Ng_sr,compensated=True,position='Position',BoxSize=Lbox)
    rr = FFTPower(mesh, mode='1d')
    Pk = rr.power
    ps = Pk['power'].real-Pk.attrs['shotnoise']

    k0,Pk = Pk['k'],ps*Pk['k']*Pk['k']*Pk['k']/(2*np.pi*np.pi)
    
    fig, ax = plt.subplots(figsize=(6,5),constrained_layout=True)
    ax.plot(k0,Pk,label='sr')
    
    k,ps = np.load(tgt_ps_path)
    ax.plot(k,ps,label='hr')  

    k,ps = np.load(lr_ps_path)
    ax.plot(k,ps,label='lr')  
    
    ax.set_xscale('log')    
    ax.set_yscale('log')
    ax.set_xlim(0.2,)
    ax.set_ylim(0.05,)
    ax.legend(fontsize=15)
    ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",fontsize=15)
    ax.set_ylabel(r"$\Delta^2(k)$",fontsize=15)
    
    return fig







