"""Quality control in three levels, after Ego2Robot: per-frame pipeline checks (checks), action statistics
(actions), and a VLM audit of the composite (vlm_audit). Curate writes L1/L2; VlmAudit writes L3 and the
final decision."""
