# core_audit/uncertainty_gated_inference.py
import torch
from typing import Dict, Optional

class UncertaintyGatedPredictor:
    """
    Gates predictions based on modality disagreement uncertainty
    Implements "know your limits" principle for high-stakes applications
    """
    def __init__(self, core_model, classifier_head, 
                 disagreement_threshold: float = 0.4,
                 fallback_policy: str = "defer_to_human"):
        self.core = core_model
        self.classifier = classifier_head.eval()
        self.disagreement_threshold = disagreement_threshold
        self.fallback_policy = fallback_policy  # "defer_to_human", "use_visual_only", "use_text_only"
        
    def predict_with_uncertainty(self,
                               image: torch.Tensor,
                               text_context: Optional[str] = None) -> Dict:
        """
        Make prediction with uncertainty-aware gating
        
        Returns: {
            "prediction": Optional[str],
            "confidence": float,
            "uncertainty_reason": Optional[str],
            "fallback_triggered": bool,
            "audit_trail": Dict  # For later review
        }
        """
        audit_trail = {}
        
        # Encode both modalities
        with torch.no_grad():
            visual_feat = self.core.clip.encode_image(
                image.unsqueeze(0).to(self.core.device)
            ).cpu().squeeze(0)
            visual_proj = self.core.projections.to_latent_space(
                visual_feat.unsqueeze(0).to(self.core.device)
            ).squeeze(0).cpu()
            
            audit_trail['visual_proj_norm'] = torch.norm(visual_proj).item()
            
            if text_context:
                text_feat = self.core.clip.encode_text(
                    self.core.tokenizer([text_context]).to(self.core.device)
                ).cpu().squeeze(0)
                text_proj = self.core.projections.to_latent_space(
                    text_feat.unsqueeze(0).to(self.core.device)
                ).squeeze(0).cpu()
                
                audit_trail['text_proj_norm'] = torch.norm(text_proj).item()
                
                # Compute modality disagreement
                disagreement = 1 - torch.cosine_similarity(
                    visual_proj.unsqueeze(0), text_proj.unsqueeze(0), dim=1
                ).item()
                audit_trail['modality_disagreement'] = disagreement
            else:
                disagreement = 0.0  # No text context = no disagreement signal
                audit_trail['modality_disagreement'] = None
        
        # Decision logic
        if disagreement > self.disagreement_threshold:
            # High uncertainty: trigger fallback
            if self.fallback_policy == "defer_to_human":
                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "uncertainty_reason": f"High modality disagreement ({disagreement:.3f} > {self.disagreement_threshold})",
                    "fallback_triggered": True,
                    "audit_trail": audit_trail,
                    "action_required": "human_review"
                }
            elif self.fallback_policy == "use_visual_only":
                # Use visual projection only, but flag reduced confidence
                logits = self.classifier(visual_proj.unsqueeze(0).to(self.core.device))
                probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
                confidence = probs.max().item() * (1 - disagreement)  # Discount by disagreement
                
                return {
                    "prediction": self._logits_to_label(logits),
                    "confidence": confidence,
                    "uncertainty_reason": f"Modality disagreement ({disagreement:.3f}); using visual-only with discounted confidence",
                    "fallback_triggered": True,
                    "audit_trail": audit_trail,
                    "warning": "Prediction based on visual modality only due to text-visual mismatch"
                }
            # Add other fallback policies as needed
        
        # Low disagreement: proceed with fused prediction
        # Simple fusion: average projections (in production: learned fusion)
        if text_context:
            fused_proj = (visual_proj + text_proj) / 2
        else:
            fused_proj = visual_proj
            
        logits = self.classifier(fused_proj.unsqueeze(0).to(self.core.device))
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        
        return {
            "prediction": self._logits_to_label(logits),
            "confidence": probs.max().item(),
            "uncertainty_reason": None,
            "fallback_triggered": False,
            "audit_trail": audit_trail,
            "modality_disagreement": disagreement
        }
    
    def _logits_to_label(self, logits: torch.Tensor) -> str:
        """Convert classifier logits to label string (stub)"""
        # In production: map index to label via dataset metadata
        return f"label_{logits.argmax().item()}"