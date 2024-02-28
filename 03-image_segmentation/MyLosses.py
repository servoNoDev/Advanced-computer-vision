import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(self):
        """
        Initializes the JaccardLoss module.
        """
        super(JaccardLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculates the Jaccard loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Computed Jaccard loss.
        """
        # Calculate the intersection of predictions and targets
        intersection = torch.sum(predictions * targets)
        # Calculate the union of predictions and targets
        union = torch.sum(predictions + targets) - intersection
        # Calculate the Jaccard index with a small epsilon to avoid division by zero
        jaccard = (intersection + 1e-5) / (union + 1e-5)
        # Calculate the Jaccard loss
        jaccard_loss = 1 - jaccard
        return jaccard_loss


class DiceLoss(nn.Module):
    def __init__(self):
        """
        Initializes the DiceLoss module.
        """
        super(DiceLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculates the Dice loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Computed Dice loss.
        """
        # Calculate the intersection of predictions and targets
        intersection = torch.sum(predictions * targets)
        # Calculate the Dice coefficient with a small epsilon to avoid division by zero
        dice_coefficient = (2.0 * intersection + 1e-5) / (torch.sum(predictions) + torch.sum(targets) + 1e-5)
        # Calculate the Dice loss
        dice_loss = 1 - dice_coefficient
        return dice_loss

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        Initializes the BinaryCrossEntropyLoss module.
        """
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculates the Binary Cross-Entropy loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Computed Binary Cross-Entropy loss.
        """
        # Apply sigmoid activation to predictions to ensure they are in the range [0, 1]
        predictions = torch.sigmoid(predictions)
        # Compute the Binary Cross-Entropy loss
        bce_loss = nn.BCELoss()(predictions, targets)
        return bce_loss
