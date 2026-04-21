import torch
import torch.nn as nn
from transformers import DetrForObjectDetection


class DETROreDetector(nn.Module):
    def __init__(self, num_classes=2, model_name="facebook/detr-resnet-50"):
        super().__init__()
        self.num_classes = num_classes
        self.model = DetrForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes + 1, ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def predict(self, images, threshold=0.5):
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=images)

        probas = outputs.logits.softmax(-1)[..., :-1]
        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            scores, labels = probas[i].max(-1)
            keep = scores > threshold
            results.append(
                {
                    "scores": scores[keep],
                    "labels": labels[keep],
                    "boxes": outputs.pred_boxes[i][keep],
                }
            )
        return results

    def get_optimizer_params(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": 1e-5,
            },
        ]
        return param_dicts
