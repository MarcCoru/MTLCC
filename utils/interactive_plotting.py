from __future__ import print_function
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np

def show_rgb(x, name=""):
    
    if len(x.shape)==5: # BTHWD
        max_b, max_t,_,_,max_d = x.shape
    elif len(x.shape)==4: # BHWD
        max_b,_,_,max_d = x.shape
    
    def norm(band):
        return (band - band.min()) / (band - band.min()).max()
    
    def _show_map_BTHWD(t,d,b): 
        plt.title("{name} RGB map {rd}-{gn}-{bl}, b={b}, t={t}".format(name=name,b=b,t=t,rd=d-1,gn=d,bl=d+1))
        plt.imshow(np.stack((norm(x[b,t,:,:,d-1]),norm(x[b,t,:,:,d]),norm(x[b,t,:,:,d+1])),axis=-1))
        
    def _show_map_BHWD(d,b): 
        plt.title("{name} RGB map {rd}-{gn}-{bl}, b={b}".format(name=name,b=b,rd=d-1,gn=d,bl=d+1))
        plt.imshow(np.stack((norm(x[b,:,:,d-1]),norm(x[b,:,:,d]),norm(x[b,:,:,d+1])),axis=-1))
        
    # both
    b_slider = widgets.IntSlider(description='batch',min=0,max=max_b-1,step=1,value=max_b/2)
    d_slider = widgets.IntSlider(description='band',min=1,max=max_d-2,step=1,value=max_d/2) 
        
    if len(x.shape)==5: # BTHWD
        t_slider = widgets.IntSlider(description='time',min=0,max=max_t-1,step=1,value=max_t/2)
        w = interactive(_show_map_BTHWD, t=t_slider, d=d_slider, b=b_slider)
        
    elif len(x.shape)==4: # BHWD
        w = interactive(_show_map_BHWD, d=d_slider, b=b_slider)
    
    w.layout.height = '400px'
    display(w)


def show_gray(x, name="",vmin=None, vmax=None):
    
    if len(x.shape)==5: # BTHWD
        max_b, max_t,_,_,max_d = x.shape
    elif len(x.shape)==4: # BHWD
        max_b,_,_,max_d = x.shape
    elif len(x.shape)==3: # BHW
        max_b,_,_ = x.shape
        
    def _show(x,title):
        plt.title(title)
        plt.imshow(x,vmax=vmax, vmin=vmin);
        plt.colorbar()
        
        
    def _show_map_BTHWD(t,d,b): 
        _show(x[b,t,:,:,d],"{name} feature map b={b}, t={t}, d={d}".format(name=name,b=b,t=t,d=d))
        
    def _show_map_BHWD(d,b): 
        _show(x[b,:,:,d],"{name} feature map b={b}, d={d}".format(name=name,b=b,d=d))
        
    def _show_map_BHW(b): 
        _show(x[b,:,:],"{name} feature map b={b}".format(name=name,b=b))
        
    # all
    b_slider = widgets.IntSlider(description='batch',min=0,max=max_b-1,step=1,value=max_b/2)
         
    if len(x.shape)==5: # BTHWD
        d_slider = widgets.IntSlider(description='band',min=0,max=max_d-1,step=1,value=max_d/2) 
        t_slider = widgets.IntSlider(description='time',min=0,max=max_t-1,step=1,value=max_t/2)
        w = interactive(_show_map_BTHWD, t=t_slider, d=d_slider, b=b_slider)
        
    elif len(x.shape)==4: # BHWD
        d_slider = widgets.IntSlider(description='band',min=0,max=max_d-1,step=1,value=max_d/2) 
        w = interactive(_show_map_BHWD, d=d_slider, b=b_slider)
        
    elif len(x.shape)==3: # BHW
        w = interactive(_show_map_BHW, b=b_slider)
    
    w.layout.height = '400px'
    display(w)
    
def show(x,name="",mode="RGB"):
    if mode=="RGB":
        show_rgb(x,name)
    elif mode=="gray":
        show_gray(x,name)
    
def norm_ptp(arr):
    return (arr-arr.min()) / (arr-arr.min()).max()


def norm_std(arr,stddev=1):
    arr -= arr.mean(axis=0).mean(axis=0)
    arr /= stddev*arr.std(axis=0).std(axis=0) # [-1,1]
    arr = (arr/2) + 0.5 # [0,1]
    arr = np.clip(arr*255,0,255) # [0,255]
    return arr.astype("uint8")

def norm_rgb(arr):
    # taken from QGIS mean +- 2 stddev over cloudfree image
    vmin = np.array([-0.0433,-0.0054,-0.0237])
    vmax = np.array([0.1756,0.1483,0.1057])

    arr-=vmin
    arr/=(vmax-vmin)

    return np.clip((arr*255),0,255).astype("uint8")

def write(arr,outfile):
    #norm_img = norm(arr)
    img = Image.fromarray(arr)
    img.save(outfile)

def dump3(array,name,outfolder,cmap="inferno",norm=norm_ptp):
    
    filenpath="{outfolder}/sample{s}/{name}/{d}.png"

    cmap = plt.get_cmap(cmap)

    # normalize over the entire array
    #array = norm(array)
    
    samples,h,w,depth = array.shape
    for s in range(samples):
        for d in range(depth):
            outfilepath = filenpath.format(outfolder=outfolder,s=s,name=name,d=d)

            if not os.path.exists(os.path.dirname(outfilepath)): 
                os.makedirs(os.path.dirname(outfilepath))
            arr = array[s,:,:,d]
            arr = cmap(arr)

            write((arr*255).astype('uint8'),outfilepath)

def dump(array,name,outfolder,cmap="inferno",norm=norm_ptp):
    
    filenpath="{outfolder}/sample{s}/time{t}/{d}_{name}.png"

    cmap = plt.get_cmap(cmap)

    # normalize over the entire array
    #array = norm(array)
    
    samples,times,h,w,depth = array.shape
    for s in range(samples):
        for t in range(times):
            for d in range(depth):
                outfilepath = filenpath.format(outfolder=outfolder,s=s,t=t,name=name,d=d)

                if not os.path.exists(os.path.dirname(outfilepath)): 
                    os.makedirs(os.path.dirname(outfilepath))
                arr = array[s,t,:,:,d]
                arr = cmap(arr)

                write((arr*255).astype('uint8'),outfilepath)
    
def dump_rgb(array,name,outfolder,stddev):
    
    filenpath="{outfolder}/sample{s}/time{t}_{name}.png"

    
    
    samples,times,h,w,depth = array.shape
    for s in range(samples):
        for t in range(times):
            outfilepath = filenpath.format(outfolder=outfolder,s=s,t=t,name=name)

            if not os.path.exists(os.path.dirname(outfilepath)): 
                os.makedirs(os.path.dirname(outfilepath))
            arr = array[s,t,:,:,0:3]
            
            arr = norm_std(arr,stddev=stddev)

            write(arr,outfilepath)
        
def dump_class(array,name,outfolder,cmap="Accent"):
    filenpath="{outfolder}/sample{s}/{name}.png"
    samples,h,w = array.shape

    array = array.astype(float) / 26
    
    cmap = plt.get_cmap(cmap)
    for s in range(samples):
        outfilepath = filenpath.format(outfolder=outfolder,s=s,name=name)


        arr = (cmap(array[s])*255).astype("uint8")
        write(arr,outfilepath)