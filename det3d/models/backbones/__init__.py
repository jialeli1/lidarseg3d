import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD
    from .scn_unet import UNetSCN3D
    from .scn_unet_cylinder3d import UNetCylinder3D

    from .polarnet_backbone import PolarNet_BEV_Unet
    from .cylinder3d_backbone import Cylinder3D_Asymm_3d_spconv, Cylinder3D_Asymm_3d_spconv_v2p
else:
    print("No spconv, sparse convolution disabled!")

