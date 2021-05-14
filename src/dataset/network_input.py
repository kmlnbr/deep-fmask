import torch


def get_inp_channels(arch_mode):
    num_inp_channels = {'all': 13,
                        'rgb': 3,
                        'nonveg': 9,
                        'nonvegndsi': 10,
                        'swirndsi': 7,
                        'swirndsindwi': 8,
                        'swirndwi': 7,
                        'allndsi': 14,
                        'nonvegndwi': 10,
                        'nonvegndvi': 10,
                        'vnir':4,
                        'swir':6,}

    return num_inp_channels[arch_mode]


def all(inp_img):
    return inp_img

def rgb(inp_img):
    return inp_img[:,1:4,:,:,]

def nonveg(inp_img):
    return inp_img[:,(0,1,2,3,7,8,9,10,11),:,:]

def vnir(inp_img):
    return inp_img[:,(1,2,3,7),:,:]

def swir(inp_img):
    return inp_img[:,(1,2,3,7,10,11),:,:]


def nonvegndsi(inp_img):
    b3 = inp_img[:,2,:,:]
    b11 = inp_img[:,10,:,:]
    deno = b11+b3
    deno[deno == 0] = 1
    ndsi = (b3 - b11) / deno
    inp_img[:,12,:,:] = ndsi
    return inp_img[:,(0,1,2,3,7,8,9,10,11,12),:,:]


def swirndsi(inp_img):
    b3 = inp_img[:,2,:,:]
    b11 = inp_img[:,10,:,:]
    deno = b11+b3
    deno[deno == 0] = 1
    ndsi = (b3 - b11) / deno
    inp_img[:,12,:,:] = ndsi
    return inp_img[:,(1,2,3,7,10,11,12),:,:]


def swirndsindwi(inp_img):
    b3 = inp_img[:,2,:,:]
    b11 = inp_img[:,10,:,:]
    b8 = inp_img[:,7,:,:]
    deno = b11+b3
    deno1 = b8+b3
    deno[deno == 0] = 1
    deno1[deno1 == 0] = 1
    ndsi = (b3 - b11) / deno
    ndwi = (b3 - b8) / deno1
    inp_img[:,12,:,:] = ndsi
    inp_img[:,0,:,:] = ndwi
    return inp_img[:,(0,1,2,3,7,10,11,12),:,:]



def swirndwi(inp_img):
    b3 = inp_img[:,2,:,:]
    b8 = inp_img[:,7,:,:]
    deno = b8+b3
    deno[deno == 0] = 1
    ndsi = (b3 - b8) / deno
    inp_img[:,12,:,:] = ndsi
    return inp_img[:,(1,2,3,7,10,11,12),:,:]



def allndsi(inp_img):
    b3 = inp_img[:,2,:,:]
    b11 = inp_img[:,10,:,:]
    deno = b11+b3
    deno[deno == 0] = 1
    ndsi = (b3 - b11) / deno
    ndsi = torch.unsqueeze(ndsi,1)

    return torch.cat((inp_img,ndsi),1)

def nonvegndwi(inp_img):
    b3 = inp_img[:,2,:,:]
    b8 = inp_img[:,7,:,:]
    deno = b8 + b3
    deno[deno == 0] = 1
    ndwi = (b3 - b8) / deno
    inp_img[:,12,:,:] = ndwi
    return inp_img[:,(0,1,2,3,7,8,9,10,11,12),:,:]

def nonvegndvi(inp_img):
    b4 = inp_img[:,3,:,:]
    b8 = inp_img[:,7,:,:]
    deno = b8 + b4
    deno[deno == 0] = 1
    ndvi = (b8 - b4) / deno
    inp_img[:,12,:,:] = ndvi
    return inp_img[:,(0,1,2,3,7,8,9,10,11,12),:,:]



def get_inp_func(arch_mode):
    inp_func = {'all': all,
                        'rgb': rgb,
                        'nonveg': nonveg,
                        'nonvegndsi': nonvegndsi,
                        'swirndsi': swirndsi,
                        'swirndsindwi': swirndsindwi,
                        'swirndwi': swirndwi,
                        'nonvegndvi': nonvegndvi,
                        'allndsi': allndsi,
                        'nonvegndwi': nonvegndwi,
                        'vnir': vnir,
                        'swir': swir,
                }

    return inp_func[arch_mode]
