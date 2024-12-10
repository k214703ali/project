import torch
import torch.nn.functional as F
import torch.nn as nn

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class AttentionPoolingLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPoolingLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        attn_weights = self.attention_weights.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, input_dim]
        attn_scores = torch.sum(x * attn_weights, dim=-1, keepdim=True)  # Shape: [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # Shape: [batch_size, seq_len, 1]
        weighted_input = x * attn_weights  # Shape: [batch_size, seq_len, input_dim]
        pooled_output = torch.sum(weighted_input, dim=1)  # Shape: [batch_size, input_dim]
        return pooled_output


class ProjectionLayerWWithoutBN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayerWWithoutBN, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, visual_features, textual_features, labels):
        device = visual_features.device
        labels = labels.to(device)

        batch_size = visual_features.size(0)
        total_loss = 0.0

        # Compute pairwise cosine similarity between visual features and textual features
        cos_sim = F.cosine_similarity(visual_features.unsqueeze(1), textual_features.unsqueeze(0), dim=-1)

        pos_loss = labels * (1 - cos_sim)  # Positive loss for similar pairs
        neg_loss = (1 - labels) * torch.clamp(cos_sim - self.margin, min=0.0)  # Negative loss for dissimilar pairs

        total_loss = torch.sum(pos_loss + neg_loss) / (batch_size * batch_size)
        return total_loss

    @staticmethod
    def create_contrastive_labels(batch_size, device):
        labels = torch.zeros(batch_size, batch_size, dtype=torch.float, device=device)
        for i in range(batch_size):
            labels[i, i] = 1  # Positive pair (same image, different view)
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
    contrastive_projection_layer_textual = ProjectionLayerWWithoutBN(input_dim=textual_features.size(-1), output_dim=visual_features.size(-1)).to(device)

    # Reshape textual features for projection
    contrastive_textual_features_reshaped = textual_features.view(batch_size * max_len, -1)
    contrastive_projected_textual_features = contrastive_projection_layer_textual(contrastive_textual_features_reshaped)
    contrastive_projected_textual_features = contrastive_projected_textual_features.view(batch_size, max_len, -1)

    # Apply attention pooling on the projected textual features
    attention_pooling_layer = AttentionPoolingLayer(input_dim=contrastive_projected_textual_features.size(-1)).to(device)
    contrastive_pooled_textual_features = attention_pooling_layer(contrastive_projected_textual_features)

    # Normalize visual and textual features
    normalized_visual_features = visual_features / torch.norm(visual_features, p=2, dim=-1, keepdim=True)
    normalized_contrastive_pooled_textual_features = contrastive_pooled_textual_features / torch.norm(contrastive_pooled_textual_features, p=2, dim=-1, keepdim=True)

    # Create contrastive labels (similar or dissimilar pairs)
    contrastive_loss_fn = ContrastiveLoss().to(device)
    labels = contrastive_loss_fn.create_contrastive_labels(batch_size, device)

    # Compute contrastive loss between the visual features and the pooled textual features
    contrastive_loss_value = contrastive_loss_fn(normalized_visual_features, normalized_contrastive_pooled_textual_features, labels)

    # Combined loss with language and contrastive components
    lambda_l = 0.01
    total_loss = language_loss + lambda_l * contrastive_loss_value

    return total_loss
