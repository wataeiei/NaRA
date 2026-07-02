# NaRA middle-layer skip smoke test

这个小验证只回答一个问题：

```text
NaRA 不给中间层创建 adapter，短跑训练是否比全层 NaRA 更快？
```

## 1. 放到 NaRA 仓库里运行

假设 NaRA 仓库在：

```text
/path/to/NaRA
```

进入 NaRA 仓库根目录后执行：

```bash
python /path/to/nara_freeze_mid_validation/patch_nara_skip_layers.py
```

这个补丁会给 `nara/tuners/nara.py` 增加一个配置项：

```yaml
skip_layer_regex: "model\\.layers\\.(?:8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23)\\."
```

匹配到的层不会创建 NaRA adapter。

## 2. 生成两份短跑配置

```bash
python /path/to/nara_freeze_mid_validation/make_smoke_configs.py \
  --base-config config/nara/llada_instruct_nara_math14k.yaml \
  --steps 200
```

会生成：

```text
config/nara/smoke_nara_full.yaml
config/nara/smoke_nara_freeze_mid.yaml
```

## 3. 跑最小对照

```bash
python train.py --config config/nara/smoke_nara_full.yaml --seed 1234
python train.py --config config/nara/smoke_nara_freeze_mid.yaml --seed 1234
```

如果 `train.py` 不识别 `max_steps`，就手动停止在相同 step，例如都停在 200 step。

## 4. 记录

只记录这几项：

```text
trainable params
peak GPU memory
100/200 steps 总耗时
平均 step time
loss 是否正常下降
```

简单判定：

```text
step time 下降 >= 15%：有初步加速信号
显存下降 >= 10%：有资源收益
loss 不 NaN 且有下降：可以进入下一轮小质量验证
```
