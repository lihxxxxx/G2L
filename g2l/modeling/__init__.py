from .g2l import G2L
ARCHITECTURES = {"G2L": G2L}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
