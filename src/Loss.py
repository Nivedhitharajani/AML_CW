import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou

class CustomLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        # Loss weights (tuned for balanced training)
        self.box = 8.0    # Increased from default 7.5 for better localization
        self.cls = 0.6    # Reduced from 0.8 to balance with focal loss
        self.dfl = 1.5    # Optimal for Distribution Focal Loss
        
        # Focal loss parameters
        self.focal_alpha = 0.8    # Balances positive/negative samples
        self.focal_gamma = 2.0    # Focuses on hard examples
        self.focal_weight = 0.3   # Controls focal loss contribution

    def forward(self, preds, batch):
        # Get default losses from parent class
        total_loss, (box_loss, cls_loss, dfl_loss) = super().forward(preds, batch)
        print("Print the pred", preds)

        
        # ---------------------------------------------------
        # Enhanced Box Loss with EIoU
        # preds[0] already contains decoded boxes (xywh format)
        pred_boxes = preds[0]  # Shape: [batch, num_anchors, 4]
        target_boxes = batch['bboxes']
        
        # Calculate EIoU (Enhanced IoU)
        iou = bbox_iou(pred_boxes, target_boxes, EIoU=True)
        box_loss = (1.0 - iou).mean() * self.box

        # ---------------------------------------------------
        # Alpha-balanced Focal Class Loss
        pred_cls = preds[1]  # Class predictions [batch, num_classes]
        target_cls = batch['cls']  # Ground truth classes
        
        # Base BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_cls, 
            target_cls, 
            reduction='none'
        )
        
        # Focal terms
        p = torch.sigmoid(pred_cls)
        pt = p * target_cls + (1-p) * (1-target_cls)
        focal_term = (1 - pt)**self.focal_gamma
        alpha_term = self.focal_alpha * target_cls + (1-self.focal_alpha)*(1-target_cls)
        
        # Combined focal loss
        focal_loss = (alpha_term * focal_term * bce_loss).mean()
        
        # ---------------------------------------------------
        # Final combined loss
        total_loss = (
            box_loss + 
            self.cls * cls_loss + 
            self.dfl * dfl_loss + 
            self.focal_weight * focal_loss
        )
        
        return total_loss, (box_loss, cls_loss, dfl_loss)
