# 文件路径: lib/ufcl_client.py

from fate.ml.nn.homo.fedavg import FedAVGClient
from lib.ufcl_trainer import UFCLTrainer

class UFCLFedAVGClient(FedAVGClient):
    """
    UFCL 版本的 FATE 客户端
    作用：在 Local Training 阶段替换默认 Trainer，启用 UFCL 策略，并无缝兼容全局异常早停
    """
    def __init__(self, 
                 ctx, 
                 model, 
                 train_set, 
                 val_set, 
                 optimizer, 
                 loss_fn, 
                 scheduler, 
                 training_args, 
                 fed_args, 
                 compute_metrics=None,
                 **kwargs):
        
        # 1. 调用父类初始化
        super().__init__(
            ctx=ctx,
            model=model,
            train_set=train_set,
            val_set=val_set,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            training_args=training_args,
            fed_args=fed_args,
            compute_metrics=compute_metrics,
            **kwargs
        )
        
        print(f">>> [UFCL Client Rank {ctx.rank}] 初始化完成，正在注入 UFCLTrainer...")
        
        # 2. 直接实例化 UFCLTrainer
        # 你的早停逻辑已经藏在 compute_metrics 里了，传进去就能生效！
        self.trainer = UFCLTrainer(
            model=model,            
            args=training_args,     
            train_dataset=train_set, 
            eval_dataset=val_set,
            compute_metrics=compute_metrics,  
            optimizers=(optimizer, scheduler) 
        )
        
        # 3. 补充可能丢失的属性引用
        if hasattr(loss_fn, 'to'):
            self.trainer.loss_func = loss_fn
            
    def train(self):
        """
        重写 train 方法，增加日志标识
        """
        if self.ctx.is_on_guest or self.ctx.is_on_host:
            print(f">>> [UFCL] 开始本地训练 (Mixup Enabled)...")
        super().train()