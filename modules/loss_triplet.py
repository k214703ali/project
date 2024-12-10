import torch
import torch.nn.functional as F
import torch.nn as nn

# Language Model Criterion (for language generation tasks, remains unchanged)
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


# Attention Pooling (unchanged, for pooling textual embeddings)
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


# Projection Layer (unchanged, for aligning text and image features)
class ProjectionLayerWWithoutBN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayerWWithoutBN, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


# New Triplet Loss Class for Visual and Textual Features
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, visual_features, textual_features, labels):
        # Normalize the visual and textual features
        visual_features = F.normalize(visual_features, p=2, dim=-1)  # (batch_size, n_features)
        textual_features = F.normalize(textual_features, p=2, dim=-1)  # (batch_size, n_features)

        # Calculate cosine similarity between anchor (visual) and positive (textual)
        positive_similarity = F.cosine_similarity(visual_features, textual_features)  # (batch_size,)

        # Generate negative samples (other patient's textual features)
        negative_similarity = F.cosine_similarity(visual_features, labels)  # labels could be negative textual embeddings

        # Compute the triplet loss: max(0, negative_similarity - positive_similarity + margin)
        loss = torch.clamp(negative_similarity - positive_similarity + self.margin, min=0.0)

        return loss.mean()


# Helper function to create labels for positive and negative pairs
def create_triplet_labels(batch_size, device, textual_features, negative_features):
    """
    Generate triplet labels: anchor, positive, negative.
    anchor = visual_features
    positive = textual_features (same patient)
    negative = negative_features (different patient)
    """
    positive_labels = textual_features  # Positive pairs (same patient)

    # Negative pairs could be randomly sampled or explicitly set to a different patient's textual embedding
    negative_labels = negative_features  # Example: textual features of a different patient

    return positive_labels, negative_labels


# Compute loss function with triplet loss for visual and textual embeddings
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

    # Generate positive and negative labels for the triplet loss
    positive_labels, negative_labels = create_triplet_labels(batch_size, device, normalized_contrastive_pooled_textual_features, fc_feats_1)  # Example: Use fc_feats_1 for negatives

    # Triplet Loss (anchor: visual, positive: textual, negative: other textual)
    triplet_loss_fn = TripletLoss(margin=1.0).to(device)
    triplet_loss_value = triplet_loss_fn(normalized_visual_features, normalized_contrastive_pooled_textual_features, negative_labels)

    # Combined loss with language and triplet components
    lambda_l = 0.01
    total_loss = language_loss + lambda_l * triplet_loss_value

    return total_loss
