from .fusion import Net3
from .loss_vif import fusion_loss_vif

MODELS = {
          "Fusion":Net3
            }

LOSSES = {
          "Loss_Vif":fusion_loss_vif
}