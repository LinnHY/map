# TeacherStudent.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from vilt.modules import ViLTransformerSS
from vilt.modules import heads, objectives, vilt_utils
class Teacher(ViLTransformerSS):
    def __init__(self, config):
        # 初始化Teacher模型
        print("Teacher_config================")
        super().__init__(config)
        self.role="teacher"

class Student(ViLTransformerSS):  
    def __init__(self, config):
        # 初始化Student模型
        print("Student_config================")
        super().__init__(config)
        self.role="student"
class TaS(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        # 传入config新建Teacher和Student
        self.teacher = Teacher(config) 
        self.student = Student(config)
        '''
        for param in self.student.transformer.parameters():
            param.requires_grad=True
        for param in self.student.text_embeddings.parameters():
            param.requires_grad=True
        for param in self.student.token_type_embeddings.parameters():
            param.requires_grad=True
        '''
        for param in self.teacher.transformer.parameters():
            param.requires_grad=True
        for param in self.teacher.text_embeddings.parameters():
            param.requires_grad=True
        for param in self.teacher.token_type_embeddings.parameters():
            param.requires_grad=True
        
    def infer(self, batch):
        # 输入处理 为什么在这里print没用？因为根本没执行这个函数 用的是map原本的infer
        ret_s=self.student.infer(batch)
        ret_t=self.teacher.infer(batch)
        ret={"ret_s":ret_s,"ret_t":ret_t}
        return ret

    def forward(self, batch):
        # Teacher和Student联合推理 根本没执行这个函数 用的是map原本的forward
        ret = dict()
        if len(self.current_tasks) == 0: # 如果当前没有设定任务,只运行推理函数infer并返回
            ret.update(self.infer(batch))
            return ret
        # Binary classification for Hateful Memes
        ret_s=dict()
        ret_t=dict()
        '''
        if "hatememes" in self.teacher.current_tasks:
            ret_t.update(objectives.compute_hatememes(self, batch))
        if "hatememes" in self.student.current_tasks:
            ret_s.update(objectives.compute_hatememes(self, batch))
        ret={"ret_s":ret_s,"ret_t":ret_t}
        '''
        return ret

    def training_step(self, batch, batch_idx):
        # 训练步骤实现
        vilt_utils.set_task(self.teacher)
        vilt_utils.set_task(self.student)
        output_t = self.teacher(batch) #就是forward的输出
        #print("output_t=",output_t)
        output_s = self.student(batch) #就是forward的输出
        #print("output_s=",output_s)
        total_loss_t = sum([v for k, v in output_t.items() if "loss" in k])
        #print("total_loss_t=",total_loss_t)
        total_loss_s = sum([v for k, v in output_s.items() if "loss" in k])
        #print("total_loss_s=",total_loss_s)
        alpha = 0.99 # EMA的平滑系数

        total_loss=alpha *total_loss_s + (1 - alpha) * total_loss_t
        #print("total_loss",total_loss)
        #total_loss=total_loss_s
        #print("training_step完毕")

        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            student_param.data = alpha * student_param.data + (1 - alpha) * teacher_param.data

        return total_loss # 本批次的总训练损失

    def training_epoch_end(self, outs):
        # 训练结束操作
        vilt_utils.epoch_wrapup_TS(self)
        #vilt_utils.epoch_wrapup(self.teacher)
        #print("validation_epoch_end完毕")

    def validation_step(self, batch, batch_idx):
        # 验证步骤
        vilt_utils.set_task(self.teacher)
        vilt_utils.set_task(self.student)
        output_t = self.teacher(batch) #就是forward的输出
        output = self.student(batch) #就是forward的输出
        #print("validation_step完毕")

    def validation_epoch_end(self, outs):
        # 验证结束操作
        vilt_utils.epoch_wrapup_TS(self)
        #print("validation_epoch_end完毕")

    def test_step(self, batch):
        # 测试步骤
        print()

    def test_epoch_end(self):
        # 测试结束操作
        print()

    def configure_optimizers(self):
        # 优化器设置
        print("configure_optimizers完毕!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return vilt_utils.set_schedule(self)