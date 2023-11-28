from .base_dataset import BaseDataset
import torch
import random, os

class HateMemesDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["hatememes_train"]
        elif split == "val":
            names = ["hatememes_dev"]
        elif split == "test":
            names = ["hatememes_test"] 

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="text",
            remove_duplicate=False,
        )
        
        # missing modality control        
        self.simulate_missing = missing_info['simulate_missing']
        missing_ratio = missing_info['ratio'][split]
        mratio = str(missing_ratio).replace('.','')
        missing_type = missing_info['type'][split]    
        both_ratio = missing_info['both_ratio']
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        # use image data to formulate missing table
        total_num = len(self.table['image'])
        
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)
            
            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio))
                #随机返回被missing的项
                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1
                    missing_index_image  = random.sample(missing_index, int(len(missing_index)*both_ratio))
                    missing_table[missing_index_image] = 2
                    
                torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table

    def __getitem__(self, index):
        # index -> pair data index
        # image_index -> image index in table
        # question_index -> plot index in texts of the given image
        image_index, question_index = self.index_mapper[index]
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!self=",self.table['caption'])
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        simulate_missing_type = 0
        if self.split == 'train' and self.simulate_missing and self.missing_table[image_index] == 0:
            simulate_missing_type = random.choice([0,1,2])
            
        image_tensor = self.get_image(index)["image"]
        
        # missing image, dummy image is all-one image
        if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
            for idx in range(len(image_tensor)):
                image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            '''
            尝试验证caption对image的补偿作用
            '''
            caption=str(self.table['caption'][image_index]).rstrip()
            text=str(self.table['text'][image_index])[2:-2]
            sentence="This meme shows "+caption+".According to'"+text+"',in terms of hostility or not,it can be classified as [MASK]"
            encoding = self.tokenizer(
                sentence, #源代码中这里是text
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )   
            text = (sentence, encoding)
        else: 
            #missing text, dummy text is ''
            if self.missing_table[image_index] == 1 or simulate_missing_type == 1:
                text = ''
                caption=str(self.table['caption'][image_index]).rstrip()
                #print("caption!!!!!!!!!!!!!!!!!!!!!!=",type(caption))
                encoding = self.tokenizer(
                    caption, #源代码中这里是text
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_special_tokens_mask=True,
                )   
                text = (caption, encoding)
            else:
                text = self.get_text(index)["text"]

        
        labels = self.table["label"][image_index].as_py()
        #print("image_index=",image_index)
        id=self.table["id"][image_index].as_py()
        caption=self.table["caption"][image_index].as_py()
        return {
            "id":id,
            "image": image_tensor,
            "text": text,
            "label": labels,
            "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
            "caption":caption
        }
