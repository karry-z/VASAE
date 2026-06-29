# Limitations

VASAE nearest-token names are geometric labels. A feature named by token `T` means the decoder direction is near token `T` in the fixed vocabulary embedding space used for anchoring.

This does not imply that the feature fully represents that token, concept, or behavior. The label is not a complete semantic explanation.

The label is not causal evidence. It does not show that changing the feature causes the model to produce the named token or behavior.

The label is not guaranteed to be context-invariant. A feature can activate in multiple contexts, and nearest-token geometry can miss context-dependent behavior.

Use nearest-token names as compact labels for inspection. Validate semantic or causal claims with separate evidence.
