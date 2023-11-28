import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from vilt.modules.dist_utils import all_gather
from vilt.modules.objectives import compute_irtr_recall
from vilt.gadgets.my_metrics import Accuracy, VQAScore, Scalar, F1_Score, AUROC, Scalar2, check


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "mmimdb":
                setattr(pl_module, f"{split}_{k}_F1_scores", F1_Score())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
            elif k == "hatememes":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_AUROC", AUROC())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
            elif k == "food101":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())       
                
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def test_ablation(pl_module, loss_name, res):
    test_ratio = pl_module.hparams.config['test_ratio']
    exp_name = pl_module.hparams.config["test_exp_name"]
    test_type = pl_module.hparams.config["test_type"]       
    records = f'missing ratio: {test_ratio}, ' + res
    record_file = f'./records/{loss_name}/{loss_name}_{exp_name}_on_missing_{test_type}'
    with open(record_file, 'a+') as f:
        f.write(records+'\n')
                
def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue
        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        elif loss_name == "hatememes":
            value2 = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value2)       
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_AUROC").compute()
            pl_module.log(f"{loss_name}/{phase}/AUROC_epoch", value)
            #print("auroc=",value)            
            getattr(pl_module, f"{phase}_{loss_name}_AUROC").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()      
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'AUROC: {0:.2f}, Accuracy: {1:.2f}'.format(100*value, 100*value2)
                test_ablation(pl_module, loss_name, res)
            
        elif loss_name == "food101":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)       
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()   
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'Accuracy: {0:.2f}'.format(100*value)
                test_ablation(pl_module, loss_name, res)            
            
        elif loss_name == "mmimdb":
            values = getattr(pl_module, f"{phase}_{loss_name}_F1_scores").compute()
            value = values[1]
            pl_module.log(f"{loss_name}/{phase}/F1_Micro_epoch", values[0])
            pl_module.log(f"{loss_name}/{phase}/F1_Macro_epoch", values[1])
            pl_module.log(f"{loss_name}/{phase}/F1_Samples_epoch", values[2])
            pl_module.log(f"{loss_name}/{phase}/F1_Weighted_epoch", values[3])
            getattr(pl_module, f"{phase}_{loss_name}_F1_scores").reset()            
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'F1-Macro: {0:.2f}, F1-Micro: {1:.2f}, F1-Weighted: {2:.2f}, F1-Sample: {3:.2f}'.format(100*values[1], 100*values[0], 100*values[2], 100*values[3])
                test_ablation(pl_module, loss_name, res)              
            
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value
    pl_module.log(f"{phase}/the_metric", the_metric)

def epoch_wrapup_TS(TaS):
    phase = "train" if TaS.teacher.training else "val"
    the_metric = 0
    for pl_module in [TaS.teacher]:
        for loss_name, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            value = 0

            if loss_name == "hatememes":
                
                value2 = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value2)       
                getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
                value = getattr(pl_module, f"{phase}_{loss_name}_AUROC").compute()
                pl_module.log(f"{loss_name}/{phase}/AUROC_epoch", value)
                #print("auroc=",value)            
                getattr(pl_module, f"{phase}_{loss_name}_AUROC").reset()
                pl_module.log(
                    f"{loss_name}/{phase}/loss_epoch",
                    getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"{phase}_{loss_name}_loss").reset()      
                
                if pl_module.hparams.config["test_exp_name"] is not None:
                    res = 'AUROC: {0:.2f}, Accuracy: {1:.2f}'.format(100*value, 100*value2)
                    test_ablation(pl_module, loss_name, res)
            the_metric += value
    TaS.log(f"{phase}/the_metric", the_metric)
    #print("the_metric=",the_metric)
def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"] # 获得学习率的值
    wd = pl_module.hparams.config["weight_decay"] # 获得权重衰减的值
    print("lr=",lr)
    print("wd=",wd)
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ] # 不需要权重衰减的参数列表
    head_names = ["vqa_classifier", "mmimdb_classifier", "food101_classifier", "hatememes_classifier", "nlvr2_classifier"] # 所有头部层的名称
    prompt_name = "prompt"
    lr_mult = pl_module.hparams.config["lr_mult"] # 学习率的倍数，用于调整不同参数组的学习率
    end_lr = pl_module.hparams.config["end_lr"] # 学习率调度的结束学习率。学习率调度器将在训练过程中将学习率从初始值逐渐降低到这个值。
    decay_power = pl_module.hparams.config["decay_power"] # 表示学习率调度的方式，可以是 "cosine" 或其他。"cosine" 学习率调度器将学习率按余弦函数降低
    optim_type = pl_module.hparams.config["optim_type"] # 所选的优化器类型

    names = [n for n, p in pl_module.named_parameters()]
    
    optimizer_grouped_parameters = [ # 将要传递给优化器的不同参数组
        {
            "params": [ # 权重衰减参数组：这些参数组包含需要进行权重衰减（L2正则化）的模型参数。通常，权重衰减会应用于模型的权重参数，以防止过拟合
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },            
        {
            "params": [ # 不需要权重衰减参数组：这些参数组包含不需要进行权重衰减的模型参数，通常是模型的偏差（bias）参数和某些归一化（normalization）参数。这些参数通常不会被 L2 正则化。
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [ # 不同学习率参数组：有时，模型的不同部分需要不同的学习率。例如，模型的头部层和底层可能需要不同的学习率，以便更好地训练。
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]
    print("tmp1=",len(optimizer_grouped_parameters[0]["params"]))
    print("tmp2=",len(optimizer_grouped_parameters[1]["params"]))
    print("tmp3=",len(optimizer_grouped_parameters[2]["params"]))
    print("tmp4=",len(optimizer_grouped_parameters[3]["params"]))
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
