## n20_k3_uniform

### Neural baselines
python run_AM.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3 test_cfg.decode_type=greedy
python run_AM.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3
python run_POMO.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3

### Search-based baselines
# skipped NeuroLKH for n=20 (only stable for N=100 so far)
python run_NeuroLKH.py policy=lkh env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=3
python run_HGS.py env=cvrp20_unf test_cfg.time_limit=8 policy_cfg.fleet_size=3 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt test_cfg.dataset_size=1000

## n20_k4_uniform

### Neural baselines
python run_AM.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4 test_cfg.decode_type=greedy
python run_AM.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4
python run_POMO.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4

### Search-based baselines
# skipped NeuroLKH for n=20 (only stable for N=100 so far)
python run_NeuroLKH.py policy=lkh env=cvrp20_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=4
python run_HGS.py env=cvrp20_unf test_cfg.time_limit=8 policy_cfg.fleet_size=4 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt test_cfg.dataset_size=1000

## n50_k6_uniform

### Neural baselines
python run_AM.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6 test_cfg.decode_type=greedy
python run_AM.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6
python run_POMO.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6

### Search-based baselines
# skipped NeuroLKH for n=50 (only stable for N=100 so far)
python run_NeuroLKH.py policy=lkh env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=6
python run_HGS.py env=cvrp50_unf test_cfg.time_limit=8 policy_cfg.fleet_size=6 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt test_cfg.dataset_size=1000

## n50_k7_uniform

### Neural baselines
python run_AM.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7 test_cfg.decode_type=greedy
python run_AM.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7
python run_POMO.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7

### Search-based baselines
# skipped NeuroLKH for n=50 (only stable for N=100 so far)
python run_NeuroLKH.py policy=lkh env=cvrp50_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=7
python run_HGS.py env=cvrp50_unf test_cfg.time_limit=8 policy_cfg.fleet_size=7 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt test_cfg.dataset_size=1000

## n60_k6_uniform

### Neural baselines
python run_AM.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6 test_cfg.decode_type=greedy
python run_AM.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6
python run_POMO.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6

### Search-based baselines
# skipped NeuroLKH for n=60 (only stable for N=100 so far)
python run_NeuroLKH.py policy=lkh env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=6
python run_HGS.py env=cvrp60_unf test_cfg.time_limit=8 policy_cfg.fleet_size=6 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt test_cfg.dataset_size=130

## n60_k7_uniform

### Neural baselines
python run_AM.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7 test_cfg.decode_type=greedy
python run_AM.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7
python run_POMO.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7

### Search-based baselines
# skipped NeuroLKH for n=60 (only stable for N=100 so far)
python run_NeuroLKH.py policy=lkh env=cvrp60_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130 env_kwargs.sampling_args.k=7
python run_HGS.py env=cvrp60_unf test_cfg.time_limit=8 policy_cfg.fleet_size=7 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt test_cfg.dataset_size=130

## n100_k9_uniform

### Neural baselines
python run_AM.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9 test_cfg.decode_type=greedy
python run_AM.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9
python run_POMO.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9

### Search-based baselines
python run_NeuroLKH.py policy=neuro_lkh env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9
python run_NeuroLKH.py policy=lkh env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=9
python run_HGS.py env=cvrp100_unf test_cfg.time_limit=8 policy_cfg.fleet_size=9 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt test_cfg.dataset_size=1000

## n100_k10_uniform

### Neural baselines
python run_AM.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10 test_cfg.decode_type=greedy
python run_AM.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10 test_cfg.decode_type=sample test_cfg.sample_size=1280
python run_BQ.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10
python run_POMO.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10 tester_cfg.pomo_size=1
python run_POMO.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10 tester_cfg.pomo_size=20
python run_PARCO.py env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10

### Search-based baselines
python run_NeuroLKH.py policy=neuro_lkh env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10
python run_NeuroLKH.py policy=lkh env=cvrp100_unf test_cfg.time_limit=8 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000 env_kwargs.sampling_args.k=10
python run_HGS.py env=cvrp100_unf test_cfg.time_limit=8 policy_cfg.fleet_size=10 data_file_path=data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt test_cfg.dataset_size=1000
