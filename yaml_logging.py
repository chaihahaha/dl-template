import traceback
import inspect
import yaml
import sys
import os
import tqdm
import datetime
os.environ['VAR_NAME']=sys.argv[1]
logger=dict() # 用字典保存代码、进程ID、配置参数、开始时间、训练时产生的数据等日志信息
logger['code']=inspect.getsource(sys.modules[__name__])
logger['pid']=os.getpid()
logger['config']=config
logger['datetime']=str(datetime.datetime.now())
logger['loss'] = []
logger['info'] = []
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
                with open('train.log','w') as f:
                    f.write(yaml.dump(logger, Dumper=yaml.CDumper)) # 使用yaml保存日志
except KeyboardInterrupt:
    print('manully stop training...')
except Exception:
    print(traceback.format_exc())
finally:
    postprocess(model) # 训练结束后处理部分，比如保存模型权重等信息到磁盘
