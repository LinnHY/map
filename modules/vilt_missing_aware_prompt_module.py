#针对 Vilt 模型中处理缺失数据和引入提示的特定模块的代码。这可能包括处理缺失图像或文本数据的方法，以及设计提示以引导模型关注特定信息的策略。
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit
import torch.nn.functional as F
import pickle
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import sys
sys.path.append('/amax/data/lhy')
import replicate
class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.text_embeddings = BertEmbeddings(bert_config) #创建了一个 BertEmbeddings 实例，用于将输入的文本编码成嵌入向量。
        self.text_embeddings.apply(objectives.init_weights) #初始化权重

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"]) #创建了一个嵌入层，用于编码标记文本和图像的类型信息。
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            ) # 表示加载在ImageNet上预训练好的权重参数的vit模型
        else: # 表示不加载预训练模型
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"]) # 配置池化后的向量长度
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):
# 
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            # 检查配置中的 max_text_len 是否与预训练模型的最大文本长度不同。如果不同，它会对位置嵌入的权重进行插值，以适应新的 max_text_len。
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"] # 隐藏层大小

        #构建了用于不同下游任务（例如 VQA、HateMemes、Food101）的分类器层，即在模型的基础上添加特定任务的分类头
        # 如果 hatememes 任务的损失数量大于0，就创建一个名为 hatememes_classifier 的分类器
        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)# 初始化hatememes_classifier的权重
        # 如果 food101 任务的损失数量大于0
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)               
        # 如果 mmimdb 任务的损失数量大于0
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)  
        # 如果 load_path 不为空并且配置中 finetune_first 为真，它会加载预训练模型的权重。这可以用来进行迁移学习，使用预训练模型在这些分类器上进行微调。
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")
        # 初始化提示张量
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1

        #########################初始化prompt###############################
        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim) # complete_prompt所有元素初始化为0
        complete_prompt[:,0:1,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1) # 在 complete_prompt 的中间位置填充1，以便学习注意力。
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,2:3,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_text_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,1:2,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_img_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)
        #########################初始化prompt###############################
        if not self.learnt_p:
            self.complete_prompt.requires_grad=False
            self.missing_text_prompt.requires_grad=False           
            self.missing_img_prompt.requires_grad=False

        print("complete_prompt=",self.complete_prompt)
        print("complete_prompt.size()=",self.complete_prompt.size())
        print("missing_img_prompt=",self.missing_img_prompt)
        print("missing_img_prompt.size()=",self.missing_img_prompt.size())
        print("missing_text_prompt=",self.missing_text_prompt)
        print("missing_text_prompt.size()=",self.missing_text_prompt.size())
        
        # 将它们的梯度更新关闭，在训练期间固定这些参数
        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False
        
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}


    def infer( #infer对一个batch的数据进行前向推理
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]
        pre_len_img=len(img)
        #print("这里是infer的batch=",batch)     
        #print("batch[id]=",batch['id'])     
        #print("batch[text]=",batch['text'])
        #print("batch[caption]=",batch['caption'])
        #print("\ntext_ids.shape=",text_ids.shape)  # 当text缺失时，text_ids=[101, 102,   0,  ...,   0,   0,   0] shape= torch.Size([16, 128]) 16为 batch_size 128是 memes的 maximum length of text inputs
        #print("text_labels=",text_labels) # text_labels全是 [-100, -100, -100,  ..., -100, -100, -100]
        #print("text_masks=",text_masks)   # text_masks全是  [1, 1, 1,  ..., 0, 0, 0]
        #print("text_embeds.shape=",text_embeds.shape) # text_embeds.shape = torch.Size([16, 128, 768]) 16为 batch_size  也就是在text_ids扩加了一个768的维度
        #print("img.shape=",img.shape) # hatememes:img.shape= torch.Size([8, 3, 608, 544]) 8为 batch_size 我们按照[13]将输入图像的短边调整为384，将长边限制在640以下，同时保持长宽比。 mmimdb:torch.Size([8, 3, 608, 384])
        #print("len(img)=",len(img))
        
        #clipcap
        '''
        caption = replicate.run(
                    "rmokady/clip_prefix_caption:9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8",
                    input={"image": img}
                )
        print(caption)
        '''
        if image_embeds is None and image_masks is None:        
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        #print("image_embeds.shape=",image_embeds.shape)       
        # hatememes:image_embeds.shape= torch.Size([8, 不固定, 768])  8为 batch_size
        
        text_embeds, image_embeds = (text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                                     image_embeds+ self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
                                    )
        #print("image_embeds.shape=",image_embeds.shape) 未发生变化
        '''  
        #############################################################################################
        with open('/amax/data/lhy/map/DMD/saved_data.pkl', 'rb') as f:
            better_features = pickle.load(f)
        better_s_v = better_features['better_s_v'] # torch.Size([46, 12, 50])
        better_s_l = better_features['better_s_l'] # torch.Size([124, 12, 50]) 
        better_c_l = better_features['better_c_l'] # torch.Size([124, 12, 50])
        better_c_v = better_features['better_c_v'] # torch.Size([46, 12, 50])
        
        print("better_s_v.shape=",better_s_v.shape)
        print("better_s_l.shape=",better_s_l.shape)
        print("better_c_l.shape=",better_c_l.shape)
        print("better_c_v.shape=",better_c_v.shape)
        #做img
        new_v = better_c_v.unsqueeze(0).repeat(pre_len_img, 1, 1, 1) # torch.Size([16, 46, 12, 50])
        new_v=new_v.reshape(pre_len_img, -1) # torch.Size([16, 46 * 12 * 50])
        linear = nn.Linear(46 * 12 * 50, 768).cuda() 
        new_v = linear(new_v) # torch.Size([16, 768])
        #print("before new_v.shape=",new_v.shape)
        new_v = new_v.unsqueeze(1).repeat(1, image_embeds.size(1), 1)
        #print("before new_v.shape=",new_v.shape)
        #做text
        #print("new_text.shape=",new_text.shape)
        #print("pre_len_img=",pre_len_img)
        #new_v=new_v.repeat(pre_len_img, 1, 1)
        #print("after new_v.shape=",new_v.shape)

        for idx in range(pre_len_img):
            #print("batch['missing_type'][idx] =",batch["missing_type"][idx])
            if batch["missing_type"][idx] == 0:
                continue        
            elif batch["missing_type"][idx] == 1:
                continue
                
                #print("self.hparams.config['per_gpu_batchsize']=",self.hparams.config['per_gpu_batchsize'])
                #print("HERE:img.shape=",img.shape)
            elif batch["missing_type"][idx] == 2:
                continue
                #image_embeds = new_v
                #print("new_img.shape=",new_v.shape)
                #text_embeds = new_text
                #print("HERE:text_embeds.shape=",text_embeds.shape)
##############################################################################################
        '''
        #print("image_embeds.shape=",image_embeds.shape)
        # instance wise missing aware prompts
        prompts = None
        for idx in range(pre_len_img):
            #print("batch['missing_type'][idx] =",batch["missing_type"][idx])
            if batch["missing_type"][idx] == 0:
                prompt = self.complete_prompt        
            elif batch["missing_type"][idx] == 1:
                f1=idx
                prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2:
                f2=idx
                prompt = self.missing_img_prompt
                
            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)
            
            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)
        
        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long() #prompt_masks.shape= torch.Size([16, 8])
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long() #prompt_masks.shape= torch.Size([16, 96])
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()
        #print("prompt_masks.shape=",prompt_masks.shape,"text_masks.shape=",text_masks.shape,"image_masks.shape=",image_masks.shape)
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1) # 两个embeds在第一个维度叠加 第0和第2维度是一样的 mmimdb:co_embeds.shape= torch.Size([16, 729, 768]) text_embeds.shape= torch.Size([16, 512, 768]) image_embeds.shape= torch.Size([16, 217, 768])
        x = co_embeds.detach()
        
        #print("co_embeds.shape=",co_embeds.shape,"text_embeds.shape=",text_embeds.shape,"image_embeds.shape=",image_embeds.shape)
        
        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                #print("self.prompt_layers.index(i)=",self.prompt_layers.index(i))
                #print("prompts.shape=",prompts.shape)
                #print("prompts=",prompts)
                #print("prompts[:,self.prompt_layers.index(i)].shape=",prompts[:,self.prompt_layers.index(i)].shape)
                #print("prompts[:,self.prompt_layers.index(i)]=",prompts[:,self.prompt_layers.index(i)])
                #print("x=",x)
                #print("x.shape=",x.shape)
                #print("text_embeds[fl]=",text_embeds[f1])
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks, 
                                   prompts=prompts[:,self.prompt_layers.index(i)], 
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]
        
        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   # 先将prompt的表示去掉,只池化text和image部分,才能得到合适的cls_feats。x中前total_prompt_len长度过滤掉,只池化后面的部分。
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)
            
        ret = {
            "text_feats": text_feats, #从transformer输出中提取出来的文本特征表示。
            "image_feats": image_feats, #从transformer输出中提取出来的图像特征表示。
            "cls_feats": cls_feats, #汇总transformer对整个输入的理解
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels, #每个图像patch的类别索引（训练时为标注的类别,推理时通常全为0）
            "image_masks": image_masks, #图像的掩码，在计算图像分类损失时,只对image_masks中为1的位置计算,以过滤掉填充的无效区域。image_labels和image_masks主要目的是处理变长图像输入,提供必要的掩码信息,为后续图像分类或分割等任务提供支持。
            "text_labels": text_labels, #文本的标签，用于mlm
            "text_ids": text_ids, #文本的id序列
            "text_masks": text_masks, #文本的掩码
            "patch_index": patch_index,
        }
        #print("infer完毕")
        return ret

    def forward(self, batch): # forward函数的作用是在训练时定义模型的训练目标和计算损失
        #print("这里是forward的batch=",batch)
        ret = dict()
        if len(self.current_tasks) == 0: # 如果当前没有设定任务,只运行推理函数infer并返回
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              
        #print("forward的ret=",ret)
        #print("forward完毕")
        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self) # 设置当前任务,根据模型的超参数配置，将权重大于等于1的任务名称存储在 current_tasks 属性中
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        #print("total_loss=",total_loss)
        #print("training_step完毕")
        return total_loss # 本批次的总训练损失

    def training_epoch_end(self, outs): # 每个训练周期结束时调用 wrapup 摘要
        vilt_utils.epoch_wrapup(self)
        #print("training_epoch_end完毕")

    def validation_step(self, batch, batch_idx): # 处理每个验证批次 由于在验证集中，我们不关心loss，只关心性能指标，所以不用算总loss
        #print("这里是validation_step的batch=",batch)
        vilt_utils.set_task(self)
        output = self(batch) #就是forward的输出
        #print("validation_step完毕")

    def validation_epoch_end(self, outs): # 在每个验证周期结束时调用，类似于 training_epoch_end，用于整理和输出验证周期内的统计信息。
        vilt_utils.epoch_wrapup(self)
        #print("validation_epoch_end完毕")

#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx): # 处理测试数据的每个批次
        vilt_utils.set_task(self)
        output = self(batch) #运行模型的前向传播，获取模型的输出
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs): # 在整个测试周期结束时调用
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self): # 用于配置优化器。它通过调用 vilt_utils.set_schedule(self) 用来设置优化器的学习率调度策略
        print("configure_optimizers完毕")
        return vilt_utils.set_schedule(self)
