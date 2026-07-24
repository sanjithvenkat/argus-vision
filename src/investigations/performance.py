import torch


class PerformanceInvestigation:
    """
    Evaluates a model on a labeled dataset and returns
    structured performance evidence.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def run(self, model, dataloader, environment_name="unknown"):
        model.eval()

        correct = 0
        total = 0
        total_confidence = 0.0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)

                confidence, predictions = probabilities.max(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_confidence += confidence.sum().item()

        accuracy = correct / total
        mean_confidence = total_confidence / total

        return {
            "environment": environment_name,
            "samples": total,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        }