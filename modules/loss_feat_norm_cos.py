import torch
import torch.nn.functional as F
import torch.nn as nn

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class AttentionPoolingLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPoolingLayer, self).__init__()
        # Initialize a learnable attention weight for each feature dimension
        self.attention_weights = nn.Parameter(torch.randn(input_dim))  # Shape: [input_dim]

    def forward(self, x):
        """
        Apply attention pooling to the input tensor.
        x: tensor of shape [batch_size, seq_len, input_dim]
        """
        # Reshape attention weights to apply them to each token in the sequence
        attn_weights = self.attention_weights.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, input_dim]
        
        # Compute attention scores: multiply the input features with the attention weights
        attn_scores = torch.sum(x * attn_weights, dim=-1, keepdim=True)  # Shape: [batch_size, seq_len, 1]

        # Normalize the attention scores across the sequence dimension (dim=1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Shape: [batch_size, seq_len, 1]

        # Apply the attention weights to the input tensor (broadcasted across batch)
        weighted_input = x * attn_weights  # Shape: [batch_size, seq_len, input_dim]

        # Sum along the sequence length dimension to get the pooled output
        pooled_output = torch.sum(weighted_input, dim=1)  # Shape: [batch_size, input_dim]

        return pooled_output




# Projection Layer to unify dimensions
class ProjectionLayerWWithoutBN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayerWWithoutBN, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        # self.bn = nn.BatchNorm1d(output_dim)  # Add BatchNorm

    def forward(self, x):
        x = self.projection(x)
        # return self.bn(x)  # Apply batch normalization after projection
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, fc_feats_0, fc_feats_1, textual_features, labels, visual_features=None):
        # visual_features is included but not used directly in loss computation
        device = fc_feats_0.device
        labels = labels.to(device)

        batch_size = fc_feats_0.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            pos_loss_0 = self.compute_pair_loss(fc_feats_0[i], textual_features[i], labels[i, i])
            pos_loss_1 = self.compute_pair_loss(fc_feats_1[i], textual_features[i], labels[i, i])

            total_loss += pos_loss_0 + pos_loss_1

            for j in range(batch_size):
                if i != j:
                    neg_loss_0 = self.compute_pair_loss(fc_feats_0[i], textual_features[j], labels[i, j])
                    neg_loss_1 = self.compute_pair_loss(fc_feats_1[i], textual_features[j], labels[i, j])
                    total_loss += neg_loss_0 + neg_loss_1

        loss = total_loss / (batch_size * batch_size)
        return loss

    def compute_pair_loss(self, feat1, feat2, label):
        cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=-1)

        pos_loss = label * (1 - cos_sim)
        neg_loss = (1 - label) * torch.clamp(cos_sim - self.margin, min=0.0)

        return pos_loss + neg_loss

    @staticmethod
    def create_contrastive_labels(batch_size, device):
        labels = torch.zeros(batch_size, batch_size, dtype=torch.float, device=device)
        for i in range(batch_size):
            labels[i, i] = 1  # Positive pair
        return labels

def compute_loss(reports_ids, reports_masks, visual_features, textual_features, image_ids, fc_feats_0, fc_feats_1):
    device = textual_features.device
    textual_features = textual_features.to(device)
    reports_ids = reports_ids.to(device)
    reports_masks = reports_masks.to(device)

    # Language Model Criterion (cross-entropy for sequence prediction)
    language_criterion = LanguageModelCriterion().to(device)
    language_loss = language_criterion(textual_features, reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    batch_size, max_len, vocab_size_plus_one = textual_features.size()

    # Projection layer to align textual features with the visual features' dimension
    contrastive_projection_layer_textual = ProjectionLayerWWithoutBN(input_dim=textual_features.size(-1), output_dim=fc_feats_0.size(-1)).to(device)

    # Reshape textual features for projection
    contrastive_textual_features_reshaped = textual_features.view(batch_size * max_len, -1)
    contrastive_projected_textual_features = contrastive_projection_layer_textual(contrastive_textual_features_reshaped)
    contrastive_projected_textual_features = contrastive_projected_textual_features.view(batch_size, max_len, -1)

    # Apply attention pooling on the projected textual features
    attention_pooling_layer = AttentionPoolingLayer(input_dim=contrastive_projected_textual_features.size(-1)).to(device)
    contrastive_pooled_textual_features = attention_pooling_layer(contrastive_projected_textual_features)



    # # Contrastive loss
    # contrastive_loss_fn = ContrastiveLoss().to(device)
    # labels = contrastive_loss_fn.create_contrastive_labels(batch_size, device)
    # contrastive_loss_value = contrastive_loss_fn(fc_feats_0, fc_feats_1, contrastive_pooled_textual_features, labels, visual_features)

    # Normalize features before passing to contrastive loss
    normalized_fc_feats_0 = fc_feats_0 / torch.norm(fc_feats_0, p=2, dim=-1, keepdim=True)
    normalized_fc_feats_1 = fc_feats_1 / torch.norm(fc_feats_1, p=2, dim=-1, keepdim=True)
    normalized_contrastive_pooled_textual_features = contrastive_pooled_textual_features / torch.norm(contrastive_pooled_textual_features, p=2, dim=-1, keepdim=True)

    # Contrastive loss
    contrastive_loss_fn = ContrastiveLoss().to(device)
    labels = contrastive_loss_fn.create_contrastive_labels(batch_size, device)
    contrastive_loss_value = contrastive_loss_fn(normalized_fc_feats_0, normalized_fc_feats_1, normalized_contrastive_pooled_textual_features, labels, visual_features)

    # Combined loss with language and contrastive components
    lambda_l = 0.01
    total_loss = language_loss + lambda_l * contrastive_loss_value

    return total_loss