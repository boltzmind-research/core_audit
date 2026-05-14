# Audit a dataset subset
auditor = ModalityDisagreementAuditor(core_model, clip_model, tokenizer)
results = auditor.audit_batch(
    images=batch_images,
    text_captions=batch_captions,
    demographic_metadata=[{"demographic_group": meta} for meta in batch_demographics]
)
print(auditor.generate_audit_report(results))