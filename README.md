# 深度学习模板代码

* `yaml_logging.py`: 使用了yaml保存日志（yaml保存日志的好处是可以随时从日志查看当前loss多高，配置的超参数，本次实验的代码，甚至能根据pid反查进程的cpu使用率等性能细节，后续处理实验数据时非常方便），并且能随时Ctrl + C停止训练并且不丢失数据

* `rl_ray_ppo.py`: 使用Ray和pytorch实现了多卡异步并行PPO强化学习训练

* `plot_utils.py`: 可以使用@packplot装饰器包装你画图的函数（函数参数需要包含保存图片路径filename），使得图片的元信息中包含可重新绘制此图片的代码和数据，防止你忘了数据在哪或者找不到画图的代码，都隐藏在图片的exif里面了

* `never_overwrite.py`: 在代码第一行`import never_overwrite`即可避免你在实验过程中不小心写入了同名文件导致原数据丢失