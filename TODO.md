# TODO

Agent-filed bug, no verbatim human prompt (RULES.md point 1 assumes one; this
branch instead quotes the issue it fixes, per the discopy#513/#514
precedent for self-contained bug+fix pairs).

- [ ] rel-int/optyx#28 — `Box.truncation` hardcodes `Dim(2)` per wire for
      every array-backed box, silently discarding `input_dims`/`output_dims`
      for `mode`/`qmode` wires; fix `Box.truncation` and
      `Box.determine_output_dimensions` to use the requested per-wire
      dimension for `mode` wires (bit/qubit stay at 2), with a loud error on
      an array/shape mismatch instead of a silently mistyped box.
