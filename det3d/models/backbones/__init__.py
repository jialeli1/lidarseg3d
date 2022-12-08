import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD
    from .scn_unet import UNetSCN3D
    from .scn_unet_cylinder3d import UNetCylinder3D
else:
    print("No spconv, sparse convolution disabled!")

