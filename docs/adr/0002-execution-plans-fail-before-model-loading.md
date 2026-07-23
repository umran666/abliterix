# Execution Plans fail before model loading

Every Steering Recipe must compile into an immutable Execution Plan before model loading or residual extraction begins. Unsupported backend, rank, steering-mode, transform, quantization, and rollback combinations fail at this seam rather than becoming a late error or silent no-op.
