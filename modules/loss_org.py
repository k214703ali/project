import torch
import torch.nn.functional as F
import torch.nn as nn

# Language Model Criterion (existing code, no changes)
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

# Projection Layer to unify dimensions
class ProjectionLayerWWithBN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayerWWithBN, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)  # Add BatchNorm

    def forward(self, x):
        x = self.projection(x)
        return self.bn(x)  # Apply batch normalization after projection


# Pooling Layer
class PoolingLayer(nn.Module):
    def __init__(self, pool_type='avg'):
        super(PoolingLayer, self).__init__()
        self.pool_type = pool_type

    def forward(self, x):
        if self.pool_type == 'avg':
            return torch.mean(x, dim=1)
        elif self.pool_type == 'max':
            return torch.max(x, dim=1)[0]
        else:
            raise ValueError("Unsupported pooling type: {}".format(self.pool_type))

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, fc_feats_0, fc_feats_1, textual_features, labels):
        # Ensure all tensors are on the same device
        device = fc_feats_0.device
        labels = labels.to(device)

        # Loop through the batch to create pairs of (fc_feats_0, fc_feats_1) with each textual feature
        batch_size = fc_feats_0.size(0)
        
        total_loss = 0.0

        for i in range(batch_size):
            # Get the corresponding positive pair for fc_feats_0[i] and textual_features[i]
            pos_loss_0 = self.compute_pair_loss(fc_feats_0[i], textual_features[i], labels[i, i])

            # Get the corresponding positive pair for fc_feats_1[i] and textual_features[i]
            pos_loss_1 = self.compute_pair_loss(fc_feats_1[i], textual_features[i], labels[i, i])

            # Add the positive losses
            total_loss += pos_loss_0 + pos_loss_1

            # For negative pairs, we will compute the loss for all other samples in the batch
            for j in range(batch_size):
                if i != j:  # Only compute for negative pairs
                    neg_loss_0 = self.compute_pair_loss(fc_feats_0[i], textual_features[j], labels[i, j])
                    neg_loss_1 = self.compute_pair_loss(fc_feats_1[i], textual_features[j], labels[i, j])
                    total_loss += neg_loss_0 + neg_loss_1

        # Average the loss over the batch
        loss = total_loss / (batch_size * batch_size)
        return loss

    def compute_pair_loss(self, feat1, feat2, label):
        """ Compute loss for a pair of features. """
        cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=-1)  # Cosine similarity

        # Positive pairs: minimize (1 - cosine similarity)
        pos_loss = label * (1 - cos_sim)  # For positive pair, we want to maximize the cosine similarity

        # Negative pairs: penalize if similarity exceeds margin
        neg_loss = (1 - label) * torch.clamp(cos_sim - self.margin, min=0.0)  # For negative pairs, apply margin

        # Total loss for the pair
        return pos_loss + neg_loss

    @staticmethod
    def create_contrastive_labels(batch_size, device):
        labels = torch.zeros(batch_size, batch_size, dtype=torch.float, device=device)  # Initialize all pairs as negative (0)
        
        # Loop through each image in the batch
        for i in range(batch_size):
            # Set positive pairs: for each i, fc_feats_0[i] and fc_feats_1[i] should both be positive with textual_features[i]
            labels[i, i] = 1  # Set fc_feats_0[i] and fc_feats_1[i] with textual_features[i] -> positive pair
        
        # Now, labels will be 1 for the same index (positive pair) and 0 for all others (negative pairs)
        return labels



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Ensure all tensors are on the same device and correct type
        target = target.to(pred.device, dtype=pred.dtype)
        target = target.unsqueeze(1).expand_as(pred)  # Expand target to match pred shape
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
        return loss.mean()


def compute_loss(reports_ids, reports_masks, visual_features, textual_features, image_ids, fc_feats_0, fc_feats_1):
    # Ensure all inputs are on the same device
    device = visual_features.device
    textual_features = textual_features.to(device)
    reports_ids = reports_ids.to(device)
    reports_masks = reports_masks.to(device)
    
    # # Projection layers for visual and textual features
    # projection_layer_visual = ProjectionLayerWWithBN(input_dim=visual_features.size(-1), output_dim=512).to(device)  # visual features
    # projection_layer_textual = ProjectionLayerWWithBN(input_dim=textual_features.size(-1), output_dim=512).to(device)  # textual features
    # pooling_layer = PoolingLayer(pool_type='avg').to(device)

    # Project visual features
    # projected_visual_features = projection_layer_visual(visual_features)

    # # Project and pool textual features
    # batch_size, max_len, vocab_size_plus_one = textual_features.size()
    # textual_features_reshaped = textual_features.view(batch_size * max_len, -1)
    # projected_textual_features = projection_layer_textual(textual_features_reshaped)
    # projected_textual_features = projected_textual_features.view(batch_size, max_len, -1)
    # pooled_textual_features = pooling_layer(projected_textual_features)

    language_criterion = LanguageModelCriterion().to(device)
    language_loss = language_criterion(textual_features, reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    # Projection and pooling for textual features for contrastive loss (matching fc_feats_0 dimension)
    batch_size, max_len, vocab_size_plus_one = textual_features.size()
    pooling_layer = PoolingLayer(pool_type='avg').to(device)
    contrastive_projection_layer_textual = ProjectionLayerWWithBN(input_dim=textual_features.size(-1), output_dim=fc_feats_0.size(-1)).to(device)  # Match the dimension with fc_feats_0
    contrastive_textual_features_reshaped = textual_features.view(batch_size * max_len, -1)
    contrastive_projected_textual_features = contrastive_projection_layer_textual(contrastive_textual_features_reshaped)
    contrastive_projected_textual_features = contrastive_projected_textual_features.view(batch_size, max_len, -1)
    contrastive_pooled_textual_features = pooling_layer(contrastive_projected_textual_features)

    # print("contrastive_pooled_textual_features.shape:",contrastive_pooled_textual_features.shape)
    # print("fc_feats_1.shape",fc_feats_1.shape)
    contrastive_loss_fn = ContrastiveLoss(margin=0.2).to(device)
    labels = contrastive_loss_fn.create_contrastive_labels(batch_size, device)  # Use batch_size, not image_ids
    contrastive_loss_value = contrastive_loss_fn(fc_feats_0, fc_feats_1, contrastive_pooled_textual_features, labels)

    lambda_l = 0.1
    total_loss = language_loss + lambda_l * contrastive_loss_value
    return total_loss
