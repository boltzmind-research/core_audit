# core_audit/relational_bias_propagation.py
import torch
from typing import Dict, List

class RelationalBiasAnalyzer:
    """
    Analyzes how bias propagates through CORE's GNN relational graph
    Identifies "bias amplification" via message passing
    """
    def __init__(self, core_model, gnn_model, device='cuda'):
        self.core = core_model.to(device)
        self.gnn = gnn_model.to(device).eval()
        self.device = device
        
        # Bias-sensitive relation types (configurable)
        self.bias_sensitive_relations = ["spatial:near", "support:on", "part_of:top_down"]
        
    def analyze_bias_propagation(self,
                                seed_kernels: List[ObjectKernel],  # Kernels with known bias concerns
                                propagation_depth: int = 2) -> Dict:
        """
        Trace how bias signals propagate through the relational graph
        
        Returns: {
            "propagation_map": {kernel_uuid: bias_score},
            "amplification_paths": List[{source, target, relation, amplification_factor}],
            "high_risk_relations": List[{relation_type, avg_amplification}]
        }
        """
        # Initialize bias scores: 1.0 for seed kernels, 0.0 otherwise
        bias_scores = {k.uuid: (1.0 if k in seed_kernels else 0.0) 
                      for k in self.core.kernel_pool.kernels}
        
        propagation_log = []
        
        # Iterative propagation via GNN-style message passing
        for depth in range(propagation_depth):
            new_scores = bias_scores.copy()
            
            for kernel in self.core.kernel_pool.kernels:
                if bias_scores[kernel.uuid] == 0:  # Skip if no incoming bias
                    continue
                    
                # Propagate to neighbors via sensitive relations
                for rel_type, neighbor_uuids in kernel.related_kernels.items():
                    if rel_type not in self.bias_sensitive_relations:
                        continue
                        
                    for neighbor_uuid in neighbor_uuids:
                        if neighbor_uuid not in bias_scores:
                            continue
                            
                        # Attenuation factor: bias decays with graph distance
                        attenuation = 0.7 ** (depth + 1)
                        contribution = bias_scores[kernel.uuid] * attenuation
                        
                        # Log propagation event
                        propagation_log.append({
                            "depth": depth + 1,
                            "source_uuid": kernel.uuid,
                            "target_uuid": neighbor_uuid,
                            "relation": rel_type,
                            "contribution": contribution,
                            "prior_score": bias_scores.get(neighbor_uuid, 0)
                        })
                        
                        # Update target score (max aggregation)
                        new_scores[neighbor_uuid] = max(
                            new_scores.get(neighbor_uuid, 0),
                            bias_scores[kernel.uuid] * attenuation
                        )
            
            bias_scores = new_scores
        
        # Identify amplification paths (where bias increased unexpectedly)
        amplification_paths = []
        for event in propagation_log:
            prior = event['prior_score']
            post = bias_scores.get(event['target_uuid'], 0)
            if post > prior + 0.1:  # Significant amplification
                amplification_paths.append({
                    "source": event['source_uuid'],
                    "target": event['target_uuid'],
                    "relation": event['relation'],
                    "amplification_factor": (post - prior) / max(0.01, prior)
                })
        
        # Aggregate by relation type
        relation_stats = {}
        for event in propagation_log:
            rel = event['relation']
            if rel not in relation_stats:
                relation_stats[rel] = []
            relation_stats[rel].append(event['contribution'])
        
        high_risk_relations = [
            {"relation_type": rel, "avg_amplification": torch.tensor(contribs).mean().item()}
            for rel, contribs in relation_stats.items()
            if torch.tensor(contribs).mean().item() > 0.15
        ]
        
        return {
            "propagation_map": bias_scores,
            "amplification_paths": sorted(amplification_paths, key=lambda x: -x['amplification_factor'])[:20],
            "high_risk_relations": sorted(high_risk_relations, key=lambda x: -x['avg_amplification']),
            "total_propagation_events": len(propagation_log)
        }
    
    def generate_mitigation_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations to reduce bias propagation"""
        recommendations = []
        
        # Recommendation 1: Attenuate high-risk relations
        for rel in analysis['high_risk_relations']:
            recommendations.append({
                "type": "edge_weight_adjustment",
                "target_relation": rel['relation_type'],
                "action": f"Reduce GNN message weight for '{rel['relation_type']}' by {min(0.5, rel['avg_amplification'] * 0.8):.1%}",
                "expected_impact": "Reduce bias amplification along this relation type"
            })
        
        # Recommendation 2: Add bias-regularization loss for high-score kernels
        high_bias_kernels = [
            uuid for uuid, score in analysis['propagation_map'].items() 
            if score > 0.6
        ]
        if high_bias_kernels:
            recommendations.append({
                "type": "regularization_loss",
                "target_kernels": high_bias_kernels[:10],  # Top 10
                "action": "Add orthogonality loss to discourage biased kernel embeddings from dominating GNN messages",
                "expected_impact": "Reduce representation collapse toward biased prototypes"
            })
        
        # Recommendation 3: Human review queue
        flagged = [
            {"uuid": path['target'], "reason": f"Amplified via {path['relation']} from {path['source']}"}
            for path in analysis['amplification_paths'][:5]
        ]
        if flagged:
            recommendations.append({
                "type": "human_review_queue",
                "flagged_kernels": flagged,
                "action": "Queue these kernels for manual bias assessment before deployment",
                "expected_impact": "Catch high-risk propagation cases before they affect downstream decisions"
            })
        
        return recommendations