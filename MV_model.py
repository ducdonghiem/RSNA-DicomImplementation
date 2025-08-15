import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class AttentionAggregator(nn.Module):
    """
    Attention mechanism for aggregating multiple feature vectors from the same view.
    Learns to assign weights to each scan based on their relevance for classification.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        """
        Args:
            feature_dim: Dimension of input feature vectors (e.g., 1280 for EfficientNetB0)
            hidden_dim: Hidden dimension for attention network
        """
        super(AttentionAggregator, self).__init__()
        
        # Attention network: learns to compute attention weights
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output single attention score per feature vector
        )
        
    def forward(self, feature_vectors):
        """
        Args:
            feature_vectors: Tensor of shape (batch_size, num_scans, feature_dim)
        
        Returns:
            aggregated_features: Tensor of shape (batch_size, feature_dim)
        """
        batch_size, num_scans, feature_dim = feature_vectors.shape
        
        # Compute attention scores for each scan
        # Reshape to (batch_size * num_scans, feature_dim) for linear layers
        reshaped_features = feature_vectors.view(-1, feature_dim)
        attention_scores = self.attention_net(reshaped_features)  # (batch_size * num_scans, 1)
        
        # Reshape back to (batch_size, num_scans, 1)
        attention_scores = attention_scores.view(batch_size, num_scans, 1)
        
        # Apply softmax to get attention weights (sum to 1 across scans)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, num_scans, 1)
        
        # Compute weighted sum of feature vectors
        aggregated_features = torch.sum(attention_weights * feature_vectors, dim=1)  # (batch_size, feature_dim)
        
        return aggregated_features


class MV_Model(nn.Module):
    """
    Multi-view model for breast cancer detection using MLO and CC views.
    Handles variable number of scans per view with attention-based aggregation.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, feature_dim: int = 1280):     # 1280 for EfficientNetB0, 768 for ConvNeXt-V1 small
        """
        Args:
            num_classes: Number of output classes (default: 2 for cancer/no cancer)
            pretrained: Whether to use pretrained EfficientNetB0 weights
            feature_dim: Feature dimension from encoders (1280 for EfficientNetB0)
        """
        super(MV_Model, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Feature encoders for MLO and CC views (shared architecture)
        self.mlo_encoder = self._create_encoder(pretrained)
        self.cc_encoder = self._create_encoder(pretrained)
        
        # Attention aggregators for each view
        self.mlo_aggregator = AttentionAggregator(feature_dim)
        self.cc_aggregator = AttentionAggregator(feature_dim)
        
        # Classifier for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # Concatenated features from both views
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    # def _create_encoder(self, pretrained: bool) -> nn.Module:
    #     """Create a feature encoder based on EfficientNetB0."""
    #     if pretrained:
    #         encoder = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    #     else:
    #         encoder = torchvision.models.efficientnet_b0(weights=None)
        
    #     # Remove the final classifier to get feature vectors
    #     # EfficientNetB0 has features + avgpool + classifier
    #     encoder = nn.Sequential(*list(encoder.children())[:-1])  # Remove classifier
        
    #     return encoder
    
    def _create_encoder(self, pretrained: bool) -> nn.Module:
        """Create a feature encoder based on EfficientNetV2-B0 (7.1M params)."""
        if pretrained:
            # weights = torchvision.models.EfficientNet_V2_B0_Weights.IMAGENET1K_V1
            # encoder = torchvision.models.efficientnet_v2_b0(weights='IMAGENET1K_V1')
            encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')  # ~21M params
        else:
            encoder = torchvision.models.efficientnet_v2_s(weights=None)
        
        # Remove the classifier to get feature vectors
        encoder = nn.Sequential(*list(encoder.children())[:-1])  # Removes classifier
        
        return encoder

    # def _create_encoder(self, pretrained: bool) -> nn.Module:
    #     """Create a feature encoder based on ConvNeXt-V1 small."""
    #     if pretrained:
    #         encoder = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
    #     else:
    #         encoder = torchvision.models.convnext_small(weights=None)
        
    #     # ConvNeXt's architecture has a features block and a classifier block.
    #     # We remove the classifier to get feature vectors.
    #     encoder = nn.Sequential(*list(encoder.children())[:-1])
        
    #     return encoder
    
    # def _extract_view_features_batch(self, scans_batch, encoder):
    #     """
    #     Extract features from multiple scans across a batch.
        
    #     Args:
    #         scans_batch: List of lists, where each inner list contains tensors for one sample
    #                     [[sample1_scan1, sample1_scan2], [sample2_scan1], ...]
    #         encoder: The encoder network for this view
            
    #     Returns:
    #         batch_features: List of tensors, each of shape (num_scans_for_sample, feature_dim)
    #     """
    #     batch_features = []
        
    #     for sample_scans in scans_batch:  # Iterate through each sample in the batch
    #         if not sample_scans:  # Handle empty list case
    #             raise ValueError("No scans provided for sample")
            
    #         sample_features = []
            
    #         for scan in sample_scans:  # Iterate through scans for this sample
    #             # Ensure scan is a tensor
    #             if not isinstance(scan, torch.Tensor):
    #                 raise ValueError(f"Expected tensor, got {type(scan)}")
                
    #             # Add batch dimension if needed (single sample processing)
    #             if scan.dim() == 3:  # (channels, height, width)
    #                 scan = scan.unsqueeze(0)  # (1, channels, height, width)
                
    #             # Extract features from this scan
    #             features = encoder(scan)  # (1, feature_dim, 1, 1)
    #             features = features.squeeze(-1).squeeze(-1).squeeze(0)  # (feature_dim,)
    #             sample_features.append(features)
            
    #         # Stack features for this sample
    #         sample_features_tensor = torch.stack(sample_features, dim=0)  # (num_scans, feature_dim)
    #         batch_features.append(sample_features_tensor)
        
    #     return batch_features
    
    # def _aggregate_batch_features(self, batch_features, aggregator):
    #     """
    #     Apply attention aggregation to a batch of variable-length feature sequences.
        
    #     Args:
    #         batch_features: List of tensors, each of shape (num_scans_for_sample, feature_dim)
    #         aggregator: AttentionAggregator instance
            
    #     Returns:
    #         aggregated_batch: Tensor of shape (batch_size, feature_dim)
    #     """
    #     aggregated_samples = []
        
    #     for sample_features in batch_features:
    #         # Add batch dimension for aggregator
    #         sample_features_batch = sample_features.unsqueeze(0)  # (1, num_scans, feature_dim)
            
    #         # Aggregate using attention
    #         aggregated_sample = aggregator(sample_features_batch)  # (1, feature_dim)
    #         aggregated_sample = aggregated_sample.squeeze(0)  # (feature_dim,)
            
    #         aggregated_samples.append(aggregated_sample)
        
    #     # Stack all samples to create final batch tensor
    #     aggregated_batch = torch.stack(aggregated_samples, dim=0)  # (batch_size, feature_dim)
        
    #     return aggregated_batch


    # Assuming your model is on the correct device (CPU/GPU)
    def _extract_view_features_batch(self, scans_batch, encoder):
        """
        Corrected method to extract features from multiple scans in a batch.
        """
        # Flatten the list of lists into a single list of all scans in the batch
        all_scans = [scan for sample_scans in scans_batch for scan in sample_scans]
        
        # Concatenate all scans into a single tensor for batch processing
        if not all_scans:
            return [torch.empty(0, self.feature_dim, device='cuda')] * len(scans_batch)
        
        all_scans_tensor = torch.stack(all_scans, dim=0).to('cuda')
        
        # Process all scans in one forward pass
        all_features_tensor = encoder(all_scans_tensor)
        
        # Squeeze the spatial dimensions
        all_features_tensor = all_features_tensor.squeeze(-1).squeeze(-1)
        
        # Re-group features back into a list of tensors for each sample
        batch_features = []
        current_idx = 0
        for sample_scans in scans_batch:
            num_scans = len(sample_scans)
            sample_features = all_features_tensor[current_idx:current_idx + num_scans]
            batch_features.append(sample_features)
            current_idx += num_scans
        
        return batch_features

    def _aggregate_batch_features(self, batch_features, aggregator):
        """
        Corrected aggregation to handle a list of tensors from the previous step.
        """
        aggregated_samples = []
        for sample_features in batch_features:
            if sample_features.shape[0] == 0:
                # Handle case with no scans gracefully, e.g., with a zero vector
                aggregated_samples.append(torch.zeros(self.feature_dim, device='cuda'))
            else:
                # Add batch dimension for aggregator, which expects (batch, num_scans, feature_dim)
                sample_features_batch = sample_features.unsqueeze(0)
                aggregated_sample = aggregator(sample_features_batch)
                aggregated_sample = aggregated_sample.squeeze(0)
                aggregated_samples.append(aggregated_sample)
        
        aggregated_batch = torch.stack(aggregated_samples, dim=0)
        return aggregated_batch
    
    def forward(self, mlo_scans, cc_scans):
        """
        Forward pass of the multi-view model.
        
        Args:
            mlo_scans: List of lists of MLO scan tensors 
                      [[sample1_mlo1, sample1_mlo2], [sample2_mlo1], ...]
            cc_scans: List of lists of CC scan tensors
                     [[sample1_cc1], [sample2_cc1, sample2_cc2], ...]
            
        Returns:
            output: Tensor of shape (batch_size, num_classes) with class probabilities
        """
        # Validate inputs
        if not isinstance(mlo_scans, list) or not isinstance(cc_scans, list):
            raise ValueError(f"Expected lists, got MLO: {type(mlo_scans)}, CC: {type(cc_scans)}")
        
        if len(mlo_scans) == 0 or len(cc_scans) == 0:
            raise ValueError(f"Empty scan lists: MLO has {len(mlo_scans)} samples, CC has {len(cc_scans)} samples")
        
        if len(mlo_scans) != len(cc_scans):
            raise ValueError(f"Batch size mismatch: MLO has {len(mlo_scans)} samples, CC has {len(cc_scans)} samples")
        
        # Extract features from MLO scans (batch processing)
        mlo_batch_features = self._extract_view_features_batch(mlo_scans, self.mlo_encoder)
        
        # Extract features from CC scans (batch processing)
        cc_batch_features = self._extract_view_features_batch(cc_scans, self.cc_encoder)
        
        # Aggregate features using attention mechanism
        aggregated_mlo = self._aggregate_batch_features(mlo_batch_features, self.mlo_aggregator)  # (batch_size, feature_dim)
        aggregated_cc = self._aggregate_batch_features(cc_batch_features, self.cc_aggregator)    # (batch_size, feature_dim)
        
        # Concatenate features from both views
        combined_features = torch.cat([aggregated_mlo, aggregated_cc], dim=1)  # (batch_size, feature_dim * 2)
        
        # Final classification
        output = self.classifier(combined_features)  # (batch_size, num_classes)
        
        return output
