import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import dnnlib
import tqdm
from torch_utils.download_util import find_inception_model

class InceptionFeatureExtractor:
    def __init__(self, device=torch.device('cuda')):
        self.device = device
        self.detector_path = find_inception_model(max_depth=3)
        self.detector_kwargs = dict(return_features=True)
        self.feature_dim = 2048
        
        with open(self.detector_path, 'rb') as f:
            self.detector_net = pickle.load(f).to(device)
        self.detector_net.eval()

    def extract_features(self, images):
        """
        Extract Inception-v3 features from images.
        
        Args:
            images: image tensor of shape [B, C, H, W]
            
        Returns:
            features: feature tensor of shape [B, 2048]
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        images = images.to(self.device)
        
        features = self.detector_net(images, **self.detector_kwargs)
        return features

def compute_inception_mse_loss(student_images, teacher_images, feature_extractor):
    """
    Compute MSE loss between Inception features of two image sets.
    
    Args:
        student_images: student-generated images, shape [B, C, H, W]
        teacher_images: teacher-generated images, shape [B, C, H, W]
        feature_extractor: InceptionFeatureExtractor instance
        
    Returns:
        loss: scalar MSE loss value
    """
    with torch.no_grad():
        teacher_features = feature_extractor.extract_features(teacher_images)
    student_features = feature_extractor.extract_features(student_images)
    
    loss = F.mse_loss(student_features, teacher_features.detach())

    return loss
