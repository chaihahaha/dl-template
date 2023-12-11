import traceback
import inspect
import yaml
import sys
import os
import tqdm
import datetime
import uuid
import code

def handle_exception(*args, **kwargs):
    vs = globals().copy()
    vs.update(locals())
    shell = code.InteractiveConsole(vs)
    sys.__excepthook__(*args, **kwargs)
    shell.interact()
    return

sys.excepthook = handle_exception

os.environ['VAR_NAME']=sys.argv[1]                       # 设置与实验相关的环境变量（如CUDA_VISIBLE_DEVICES）
experiment_id = str(uuid.uuid1())[:8]                    # 生成本次实验的UUID
experiment_name = name                                   # 设置描述本次实验的名称
logger=dict()                                            # 用字典保存代码、进程ID、配置参数、开始时间、训练时产生的数据等日志信息
logger['experiment_id'] = experiment_id                  # 保存本次实验的UUID
logger['experiment_name'] = experiment_name              # 保存本次实验的名称
logger['code']=inspect.getsource(sys.modules[__name__])  # 保存本次实验代码
logger['pid']=os.getpid()                                # 保存本次实验进程PID
logger['config']=config                                  # 保存配置参数
logger['datetime']=str(datetime.datetime.now())          # 保存训练开始时间
logger['loss'] = []                                      # 保存loss日志
logger['info'] = []                                      # 保存其他日志信息
logger['env_vars'] = os.environ                          # 保存相关环境变量
batch_cnt = 0
log_freq = 100
try:
    for i in tqdm.tqdm(range(N)):
        for x,y in dataset:
            loss=model.fit(x, y) # 反向传播
            logger['loss'].append(loss)
            logger['info'].append(info)
            batch_cnt += 1
            if batch_cnt % log_freq == 0: # 每log_freq个batch保存一次日志
                with open(experiment_name + experiment_id + '.log','w') as f:
                    f.write(yaml.dump(logger, Dumper=yaml.CDumper)) # 使用yaml保存日志
except KeyboardInterrupt:
    print('manully stop training...')
except Exception:
    print(traceback.format_exc())
finally:
    postprocess(model) # 训练结束后处理部分，比如保存模型权重等信息到磁盘
