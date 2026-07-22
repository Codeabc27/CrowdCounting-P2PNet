import sys, os

sys.path.insert(0, os.path.join(os.getcwd(), "CrowdCounting-P2PNet"))
from models import build_model


class Args:
    pass


a = Args()
a.backbone = "vgg16_bn"
a.row = 2
a.line = 2
print("Building model...")    
model = build_model(a)
print("Built model type:", type(model))
