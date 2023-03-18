# 深度学习模板代码

* `yaml_logging.py`: 使用了yaml保存日志（yaml保存日志的好处是可以随时从日志查看当前loss多高，配置的超参数，本次实验的代码，甚至能根据pid反查进程的cpu使用率等性能细节，后续处理实验数据时非常方便），并且能随时Ctrl + C停止训练并且不丢失数据

* `rl_ray_ppo.py`: 使用Ray和pytorch实现了多卡异步并行PPO强化学习训练
