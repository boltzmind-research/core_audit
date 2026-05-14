# core_audit/modality_disagreement.py
import torch
import numpy as np
from typing import Dict, List, Tuple

class ModalityDisagreementAuditor:
    """
    Detects bias by measuring divergence between text and visual projections
    in CORE's shared latent space
    """
    def __init__(self, core_model, clip_model, tokenizer, device='cuda'):
        self.core = core_model.to(device)
        self.clip = clip_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Thresholds for flagging disagreement
        self.disagreement_threshold = 0.4  # Cosine distance threshold
        self.min_samples_per_group = 50  # For statistical reliability
        
    def audit_batch(self,
                   images: torch.Tensor,
                   text_captions: List[str],
                   demographic_meta List[Dict]) -> Dict:
        """
        Audit a batch of multimodal samples for modality disagreement
        
        Returns: {
            "overall_disagreement": float,
            "group_disagreements": {demographic_group: float},
            "flagged_samples": List[{index, text_proj, visual_proj, distance}]
        }
        """
        B = images.size(0)
        text_projections, visual_projections = [], []
        
        # Project text and visual inputs to CORE latent space
        with torch.no_grad():
            # Visual projection
            visual_feats = self.clip.encode_image(images.to(self.device)).cpu()
            visual_projs = self.core.projections.to_latent_space(
                visual_feats.to(self.device)
            ).cpu()
            
            # Text projection
            for caption in text_captions:
                text_feat = self.clip.encode_text(
                    self.tokenizer([caption]).to(self.device)
                ).cpu().squeeze(0)
                text_proj = self.core.projections.to_latent_space(
                    text_feat.unsqueeze(0).to(self.device)
                ).squeeze(0).cpu()
                text_projections.append(text_proj)
        
        text_projections = torch.stack(text_projections)  # [B, 256]
        
        # Compute cosine distance per sample
        cos_sim = torch.cosine_similarity(text_projections, visual_projections, dim=1)
        distances = 1 - cos_sim  # Convert to distance
        
        # Aggregate by demographic group
        group_disagreements = {}
        flagged_samples = []
        
        for i in range(B):
            group = demographic_metadata[i].get('demographic_group', 'unknown')
            
            # Track per-group statistics
            if group not in group_disagreements:
                group_disagreements[group] = []
            group_disagreements[group].append(distances[i].item())
            
            # Flag high-disagreement samples
            if distances[i].item() > self.disagreement_threshold:
                flagged_samples.append({
                    "index": i,
                    "text_caption": text_captions[i],
                    "demographic_group": demographic_metadata[i].get('demographic_group'),
                    "disagreement_distance": distances[i].item(),
                    "text_proj_norm": torch.norm(text_projections[i]).item(),
                    "visual_proj_norm": torch.norm(visual_projections[i]).item()
                })
        
        # Compute group-level statistics
        group_stats = {}
        for group, dists in group_disagreements.items():
            if len(dists) >= self.min_samples_per_group:
                group_stats[group] = {
                    "mean_disagreement": np.mean(dists),
                    "std_disagreement": np.std(dists),
                    "sample_count": len(dists),
                    "flagged_ratio": sum(1 for d in dists if d > self.disagreement_threshold) / len(dists)
                }
        
        return {
            "overall_disagreement": distances.mean().item(),
            "group_disagreements": group_stats,
            "flagged_samples": flagged_samples[:100],  # Limit for review
            "audit_timestamp": time.time()
        }
    
    def generate_audit_report(self, audit_results: Dict) -> str:
        """Generate human-readable audit report"""
        report = []
        report.append(f"Multimodal Bias Audit Report")
        report.append(f"Overall modality disagreement: {audit_results['overall_disagreement']:.3f}")
        report.append(f"Samples flagged for review: {len(audit_results['flagged_samples'])}")
        report.append("\nGroup-level statistics:")
        
        for group, stats in audit_results['group_disagreements'].items():
            report.append(f"  {group}:")
            report.append(f"    Mean disagreement: {stats['mean_disagreement']:.3f} ± {stats['std_disagreement']:.3f}")
            report.append(f"    Flagged ratio: {stats['flagged_ratio']:.1%} (n={stats['sample_count']})")
        
        if audit_results['flagged_samples']:
            report.append("\nTop flagged samples (for manual review):")
            for sample in audit_results['flagged_samples'][:5]:
                report.append(f"  - Caption: '{sample['text_caption']}'")
                report.append(f"    Group: {sample['demographic_group']}, Disagreement: {sample['disagreement_distance']:.3f}")
        
        return "\n".join(report)


