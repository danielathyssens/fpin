# Drop the ORIGINAL Thyssens-2022 "old PIM" code base here (softassign perm-invariant net).
# Expected to provide, at minimum:
#   - the model class (old VRPModel.VRP_Net) producing (vrp_probs, vrp_loads, sample_path)
#   - the CVRPInstance -> model-input adapter (preprocess_PIM / transform_X2d_old)
#   - greedy decoding + capacity repair (get_path: greedy_path / make_valid)
# The original training/eval lived in a Jupyter notebook; the runnable wiring lives in
# ../pimold.py and ../runner.py, driven by ../../../run_PIMold.py.
