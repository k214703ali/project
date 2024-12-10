import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        # Ensure all tensors are on the same device (move images to correct device)
        device = images.device
        
        att_feats, fc_feats = self.visual_extractor(images.to(device))
        # Process the images to get features
        # att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0].to(device))
        # att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1].to(device))
        
        # # Concatenate the features from both images (for multi-image datasets like iu_xray)
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        
        # # For training, we pass the features through the encoder-decoder
        # if mode == 'train':
        #     output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        #     return output, fc_feats, fc_feats_0, fc_feats_1  # Return both original output and projected features

        if mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        # Ensure all tensors are on the same device (move images to correct device)
        device = images.device
        
        # Extract the features from the image
        att_feats, fc_feats = self.visual_extractor(images.to(device))
        
        # For training, we pass the features through the encoder-decoder
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            # Get the last dimension of the output (textual features) and visual features (fc_feats)
            output_dim = output.size(-1)  # Textual features dimension
            visual_dim = fc_feats.size(-1)  # Visual features dimension

            # Ensure that the projection layer and tensors are on the same device
            output = output.to(device)
            fc_feats = fc_feats.to(device)

            # Handle projection based on dimensionality comparison
            if output_dim != visual_dim:
                if output_dim > visual_dim:
                    # If textual features are larger, project visual features to match textual size
                    visual_projection = nn.Linear(visual_dim, output_dim).to(device)  # Project visual to textual
                    projected_visual_features = visual_projection(fc_feats)  # Apply projection
                    return output, projected_visual_features, fc_feats  # Return projected visual features
                else:
                    # If visual features are larger, project textual features to match visual size
                    textual_projection = nn.Linear(output_dim, visual_dim).to(device)  # Project textual to visual
                    projected_textual_features = textual_projection(output)  # Apply projection
                    return output, projected_textual_features, fc_feats  # Return projected textual features
            else:
                # If dimensions match, no projection is needed
                return output, output, fc_feats  # Return the original features without projection

        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
