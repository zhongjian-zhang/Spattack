# Spattack

Source code for AAAI 2025 paper "**Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspective**" Paper link: https://arxiv.org/pdf/2501.03301

![QQ_1736335827472](https://img.dreamcodecity.cn/img/QQ_1736335827472.png)

## 🔬 Experiment

### Spattack-O-D

```sh
python main.py --attack Spattack_O --clients_limit 0.052631 --defense Mean
```

### Spattack-O-S

```sh
python main.py --attack Spattack_O --sample_items --clients_limit 0.052631 --defense Mean
```

### Spattack-L-D

```sh
python main.py --attack Spattack_L --clients_limit 0.052631 --defense Mean
```

### Spattack-L-S

```sh
python main.py --attack Spattack_L --sample_items --clients_limit 0.052631 --defense Mean
```

You can specific the "--defense" and "--dataset" to change the defense and dataset, respectively.



## 📝 Citation and Reference

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```

```