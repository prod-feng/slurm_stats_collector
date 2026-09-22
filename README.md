# slurm_stats_collector

```
Collect Slurm cluster statistics for compute nodes and write them to YAML.

options:
  -h, --help           show this help message and exit
  -o, --output OUTPUT  Output YAML file (default: slurm_stats.yaml)

```

The YAML file looks like:

```

node_states:
  idle: 0
  mixed: 18
  allocated: 1
  draining: 0
  down: 0
cpu:
  total: 1344
  allocated: 842
  idle: 502
  down: 0
gpu:
  total: 92
  allocated: 90
  idle: 2
  down: 0
  types:
    '6000':
      total: 44
      allocated: 44
      idle: 0
      down: 0
    h200:
      total: 48
      allocated: 46
      idle: 2
      down: 0


```
