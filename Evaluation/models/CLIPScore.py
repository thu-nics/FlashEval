import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
import os

class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)


    def score(self, prompt, image_path):
        
        if (os.path.isdir(image_path)):
            indices, rewards = self.inference_rank(prompt, image_path)
            return indices, rewards
            
        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))
        
        # image encode
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))
        
        # score
        rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)
        
        return None, rewards.detach().cpu().numpy().item()

    def score_batch(self, prompts, image_paths):
        
        # if (type(image_paths).__name__=='list'):
        #     _, rewards = self.inference_rank(prompts, image_paths)
        #     return rewards
            
        # text encode
        text = clip.tokenize(prompts, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))
        
        # image encode
        for i, image_path in enumerate(image_paths):
            pil_image = Image.open(image_path)
            image1 = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            if i == 0:
                image = image1
            else:
                image = torch.cat([image, image1], dim=0)
        image_features = F.normalize(self.clip_model.encode_image(image))
        
        # score
        rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)
        
        return rewards.detach().cpu().numpy()

    def inference_rank(self, prompt, generations_list):
        
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_feature = F.normalize(self.clip_model.encode_text(text))
        
        txt_set = []
        img_set = []
        for img_name in os.listdir(generations_list):
            # image encode
            img_path = os.path.join(generations_list, img_name)
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image))
            img_set.append(image_features)
            txt_set.append(txt_feature)
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()


    def prompt_feature(self, prompt):
        with torch.no_grad():
            text = clip.tokenize(prompt, truncate=True).to(self.device)
            txt_feature = F.normalize(self.clip_model.encode_text(text))
            del text
        
        return txt_feature

    def img_feature(self, img_path):
        with torch.no_grad():
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image))
        
        return image_features