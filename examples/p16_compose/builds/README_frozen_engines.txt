These three .exe files are the FROZEN engines the P16 composition proof depends on.
tools/p16_compose.py takes them via --old / --bench and does NOT rebuild them, so
deleting them would make the BEFORE/AFTER comparison unverifiable until the task-8
revision is rebuilt -- and that revision is frozen history.

SIZES ARE AS MEASURED (getsize), not as intended:

  FROZEN_task8_bench_engine.exe    80384 B   task-8 revision, --features bench
                                             (the BEFORE: y is k-invariant, nnz pinned at 24)
  task11_bench_engine.exe          83968 B   task-11 revision, --features bench
                                             (the AFTER: k drives y, per-row terms == k)
  task11_train_control_engine.exe  96256 B   task-11 + --features train
                                             (positive control: +21504 B, proves the size
                                              apparatus reports real deltas)

NOTE ON SIZES: these are the BENCH-feature builds, so they are larger than the
default-engine budget figures quoted in docs/P16_report.md (default = 74752 B).
Only the DEFAULT build is the budgeted artifact; the bench builds are measurement
harnesses and are not budget-controlled.

Reproduce the composition acceptance with:
  python tools/p16_compose.py --bench examples/p16_compose/builds/task11_bench_engine.exe ^
      --old examples/p16_compose/builds/FROZEN_task8_bench_engine.exe ^
      --asm target/release/vmlm-asm.exe
Expected: 30 checks, 0 failed (exit 0).

The surrounding cargo target/ caches under examples/*/builds were deleted to
reclaim disk. They were scratch state, not deliverables, and are rebuildable.
These executables are the part that is NOT cheaply rebuildable.
