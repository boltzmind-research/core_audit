# core_audit/bias_aware_training.py
import torch
from typing import Dict, List

class BiasAwareTrainer:
    """
    Adjusts sample weights during training based on modality disagreement
    Reduces influence of potentially biased or mislabeled samples
    """
    def __init__(self, core_model, base_loss_fn, 
                 disagreement_weight_fn = None,
                 device='cuda'):
        self.core = core_model.to(device)
        self.base_loss = base_loss_fn
        self.device = device
        
        # Default weighting: exponential decay with disagreement
        self.disagreement_weight_fn = disagreement_weight_fn or (
            lambda d: torch.exp(-3.0 * torch.tensor(d))  # High disagreement → low weight
        )
        
        # Tracking for audit
        self.weight_statistics = []
        
    def compute_weighted_loss(self,
                            batch: Dict,
                            model_output: torch.Tensor,
                            targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with bias-aware sample weighting
        
        batch should include:
          - images: [B, C, H, W]
          - text_captions: List[str] (optional)
          - sample_meta List[Dict] with demographic info
        """
        B = batch['images'].size(0)
        sample_weights = torch.ones(B, device=self.device)
        
        # Compute modality disagreement per sample
        if 'text_captions' in batch and batch['text_captions']:
            with torch.no_grad():
                visual_feats = self.core.clip.encode_image(batch['images'].to(self.device))
                visual_projs = self.core.projections.to_latent_space(visual_feats)
                
                text_projs = []
                for caption in batch['text_captions']:
                    text_feat = self.core.clip.encode_text(
                        self.core.tokenizer([caption]).to(self.device)
                    )
                    text_proj = self.core.projections.to_latent_space(text_feat)
                    text_projs.append(text_proj.squeeze(0))
                
                text_projs = torch.stack(text_projs)
                
                # Cosine distance as disagreement signal
                cos_sim = torch.cosine_similarity(visual_projs, text_projs, dim=1)
                disagreements = 1 - cos_sim  # [B]
                
                # Map disagreement to weight
                sample_weights = self.disagreement_weight_fn(disagreements).to(self.device)
                
                # Optional: further down-weight samples from over-represented groups
                if 'sample_metadata' in batch:
                    group_counts = {}
                    for meta in batch['sample_metadata']:
                        group = meta.get('demographic_group', 'unknown')
                        group_counts[group] = group_counts.get(group, 0) + 1
                    
                    # Inverse frequency weighting (optional)
                    for i, meta in enumerate(batch['sample_metadata']):
                        group = meta.get('demographic_group', 'unknown')
                        if group_counts[group] > 100:  # Arbitrary threshold
                            sample_weights[i] *= 0.8  # Slight down-weight
        
        # Compute base loss
        base_loss = self.base_loss(model_output, targets)  # [B]
        
        # Apply weights
        weighted_loss = (sample_weights * base_loss).mean()
        
        # Log statistics for audit
        self.weight_statistics.append({
            "mean_weight": sample_weights.mean().item(),
            "min_weight": sample_weights.min().item(),
            "max_weight": sample_weights.max().item(),
            "high_disagreement_ratio": (sample_weights < 0.3).float().mean().item()
        })
        
        return weighted_loss
    
    def get_training_audit_report(self) -> Dict:
        """Generate report on how bias-aware weighting affected training"""
        if not self.weight_statistics:
            return {"status": "no_data"}
        
        import numpy as np
        weights = [s['mean_weight'] for s in self.weight_statistics]
        high_disagreement = [s['high_disagreement_ratio'] for s in self.weight_statistics]
        
        return {
            "training_steps_analyzed": len(self.weight_statistics),
            "mean_sample_weight_over_training": np.mean(weights),
            "weight_variance": np.std(weights),
            "avg_high_disagreement_ratio": np.mean(high_disagreement),
            "interpretation": (
                "Lower mean weights indicate the model is down-weighting "
                "samples with high modality disagreement, potentially reducing "
                "the influence of biased or mislabeled examples."
            )
        }