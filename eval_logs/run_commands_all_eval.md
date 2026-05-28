## Merged evaluation commands

## F-PIN | n20_k3_uniform (dist=uniform, n=20, eval_k=3, model_k=3, ds=1000)

# ### F-PIN | time_limit=5
# 
# # split decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # split decoder | add_ls=True
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=True number_runs=1
# 
# # assignment decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # assignment decoder | add_ls=True
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=True number_runs=1

### F-PIN | time_limit=8

# split decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=False number_runs=1

# split decoder | add_ls=True
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=True number_runs=1

# assignment decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=False number_runs=1

# assignment decoder | add_ls=True
python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=3 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt test_cfg.add_ls=True number_runs=1


## F-PIN | n20_k4_uniform (dist=uniform, n=20, eval_k=4, model_k=4, ds=1000)

# ### F-PIN | time_limit=5
# 
# # split decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # split decoder | add_ls=True
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=True number_runs=1
# 
# # assignment decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # assignment decoder | add_ls=True
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=True number_runs=1

### F-PIN | time_limit=8

# split decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=False number_runs=1

# split decoder | add_ls=True
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=True number_runs=1

# assignment decoder | add_ls=False
# python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=False number_runs=1

# assignment decoder | add_ls=True
python run_PIM.py env=cvrp20_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=4 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt test_cfg.add_ls=True number_runs=1


## F-PIN | n50_k6_uniform (dist=uniform, n=50, eval_k=6, model_k=6, ds=1000)

# ### F-PIN | time_limit=5
# 
# # split decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # split decoder | add_ls=True
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=True number_runs=1
# 
# # assignment decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # assignment decoder | add_ls=True
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=True number_runs=1

### F-PIN | time_limit=8

# split decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=False number_runs=1

# split decoder | add_ls=True
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=True number_runs=1

# assignment decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=False number_runs=1

# assignment decoder | add_ls=True
python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=6 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt test_cfg.add_ls=True number_runs=1


## F-PIN | n50_k7_uniform (dist=uniform, n=50, eval_k=7, model_k=7, ds=1000)

# ### F-PIN | time_limit=5
# 
# # split decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # split decoder | add_ls=True
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=True number_runs=1
# 
# # assignment decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=False number_runs=1
# 
# # assignment decoder | add_ls=True
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=5 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=True number_runs=1

### F-PIN | time_limit=8

# split decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=False number_runs=1

# split decoder | add_ls=True
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=True eval_opts_cfg.decode_vehicle_assignment=False test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=True number_runs=1

# assignment decoder | add_ls=False
# python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=False number_runs=1

# assignment decoder | add_ls=True
python run_PIM.py env=cvrp50_unf test_cfg.time_limit=8 eval_opts_cfg.post_process=False model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.use_attn=True eval_opts_cfg.nr_vehicles_eval=7 eval_opts_cfg.giant_tour_split=False eval_opts_cfg.decode_vehicle_assignment=True test_cfg.dataset_size=1000 data_file_path=data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt checkpoint_load_path=models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt test_cfg.add_ls=True number_runs=1