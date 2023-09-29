import torch
import torch.nn as nn
from torchvision import models
import wandb
import torch.nn.functional as F


def moco_contrastive_loss(q, k, T):
    """
    Compute the contrastive loss between query (q) and key (k) representations.

    Parameters:
    - q: Query representations.
    - k: Key representations.
    - T: Temperature scaling factor.

    Returns:
    - Loss value.
    """
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    logits = torch.mm(q, k.T.detach()) / T
    labels = torch.arange(logits.shape[0], dtype=torch.long).to(q.device)
    return nn.CrossEntropyLoss()(logits, labels), logits, labels


def info_nce_loss(q, k, T):
    features = torch.cat([q, k], dim=0)
    batch_size = features.shape[0] // 2  # 2 views per batch

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / T
    loss = torch.nn.CrossEntropyLoss().to(device)(logits, labels)
    return loss, logits, labels


@torch.no_grad()
def momentum_update_key_encoder(base_q, base_k, head_intra_q, head_intra_k, m):
    """
    Update the momentum of the key encoder based on the query encoder. This ensures the key encoder
    gradually aligns with the query encoder over time without direct backpropagation.

    Parameters:
    - base_q: Base of the query encoder.
    - base_k: Base of the key encoder.
    - head_intra_q: Intra-modality head of the query encoder.
    - head_intra_k: Intra-modality head of the key encoder.
    - m: Momentum factor.
    """
    for param_q, param_k in zip(base_q.parameters(), base_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
    for param_q, param_k in zip(head_intra_q.parameters(), head_intra_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


class MultiModalMoCo(nn.Module):
    """
    Multi-modal Momentum Contrast (MoCo) model for vision and tactile modalities.
    This model leverages contrastive learning between these two modalities to learn
    meaningful representations.
    """

    def __init__(self, n_channels=3, m=0.99, T=1.0, nn_model=None, intra_dim=128, inter_dim=128, weight_inter_tv=1,
                 weight_inter_vt=1, weight_intra_vision=1, weight_intra_tactile=1, pretrained_encoder=False):
        """
        Initialize the MultiModalMoCo model.

        Parameters:
        - m: Momentum factor for key encoder updates.
        - T: Temperature scaling factor for the contrastive loss.
        - nn_model: Neural network model type (e.g., 'resnet18', 'resnet50').
        - intra_dim: Dimension of the intra-modality representations.
        - inter_dim: Dimension of the inter-modality representations.
        - weight_inter: Weighting factor for inter-modality contrastive loss.
        - pretrained_encoder: Whether to use a pretrained encoder.
        """
        super(MultiModalMoCo, self).__init__()
        self.m = m
        self.T = T
        self.nn_model = nn_model
        self.intra_dim = intra_dim
        self.inter_dim = inter_dim
        self.weight_inter_tac_vis = weight_inter_tv
        self.weight_inter_vis_tac = weight_inter_vt
        self.weight_intra_vision = weight_intra_vision
        self.weight_intra_tactile = weight_intra_tactile
        self.pretrained_encoder = pretrained_encoder

        # Define vision modality encoders
        self.vision_base_q, self.vision_head_intra_q, self.vision_head_inter_q = self.create_encoder(n_channels=3)
        self.vision_base_k, self.vision_head_intra_k, self.vision_head_inter_k = self.create_encoder(n_channels=3)

        # Define tactile modality encoders
        self.tactile_base_q, self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_encoder(n_channels)
        self.tactile_base_k, self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_encoder(n_channels)

        # Initialize key encoders with query encoder weights
        momentum_update_key_encoder(self.vision_base_q, self.vision_base_k, self.vision_head_intra_q,
                                    self.vision_head_intra_k, self.m)
        momentum_update_key_encoder(self.tactile_base_q, self.tactile_base_k, self.tactile_head_intra_q,
                                    self.tactile_head_intra_k, self.m)

    def create_encoder(self, n_channels):
        """
        Create the encoder (base and heads) based on the specified neural network model.

        Returns:
        - base: Base encoder (usually a deep CNN).
        - head_intra: MLP head for intra-modality representations.
        - head_inter: MLP head for inter-modality representations.
        """

        def create_mlp_head(output_dim):
            """Create an MLP head for the given output dimension."""
            if self.nn_model == 'resnet18':
                return nn.Sequential(
                    nn.Linear(512, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, output_dim)
                )
            elif self.nn_model == 'resnet50':
                return nn.Sequential(
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, output_dim)
                )

        def create_resnet_encoder(n_channels):
            """Create a ResNet encoder based on the specified model type."""
            if self.nn_model == 'resnet18':
                resnet = models.resnet18(pretrained=self.pretrained_encoder)
            elif self.nn_model == 'resnet50':
                resnet = models.resnet50(pretrained=self.pretrained_encoder)
            if n_channels != 3:
                resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            features = list(resnet.children())[:-2]
            features.append(nn.AdaptiveAvgPool2d((1, 1)))
            features.append(nn.Flatten())
            return nn.Sequential(*features)

        base = create_resnet_encoder(n_channels)
        head_intra = create_mlp_head(self.intra_dim)
        head_inter = create_mlp_head(self.inter_dim)
        return base, head_intra, head_inter

    def forward(self, x_vision_q, x_vision_k, x_tactile_q, x_tactile_k):
        """
        Forward pass for the multi-modal MoCo model. This involves passing the inputs
        through the query and key encoders, and computing the contrastive loss.

        Parameters:
        - x_vision_q: Vision modality input for the query encoder.
        - x_vision_k: Vision modality input for the key encoder.
        - x_tactile_q: Tactile modality input for the query encoder.
        - x_tactile_k: Tactile modality input for the key encoder.

        Returns:
        - combined_loss: Combined contrastive loss value.
        """
        vision_base_q = self.vision_base_q(x_vision_q)
        vis_queries_intra = self.vision_head_intra_q(vision_base_q)
        vis_queries_inter = self.vision_head_inter_q(vision_base_q)

        # Use no_grad context for the key encoders to prevent gradient updates
        with torch.no_grad():
            vision_base_k = self.vision_base_k(x_vision_k)
            vis_keys_intra = self.vision_head_intra_k(vision_base_k)
            vis_keys_inter = self.vision_head_inter_k(vision_base_k)

        tactile_base_q = self.tactile_base_q(x_tactile_q)
        tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)
        tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)

        with torch.no_grad():
            tactile_base_k = self.tactile_base_k(x_tactile_k)
            tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
            tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)

        # Compute the contrastive loss for each pair of queries and keys
        vis_loss_intra, logits_vis_intra, labels_vis_intra = moco_contrastive_loss(vis_queries_intra, vis_keys_intra,
                                                                                   self.T)
        tac_loss_intra, logits_tact_intra, labels_tac_intra = moco_contrastive_loss(tac_queries_intra, tac_keys_intra,
                                                                                    self.T)
        vis_tac_inter, logits_vis_tac_inter, labels_vision_tactile_inter = moco_contrastive_loss(vis_queries_inter,
                                                                                                 tac_keys_inter, self.T)
        tac_vis_inter, logits_tac_vis_inter, labels_tactile_vision_inter = moco_contrastive_loss(tac_queries_inter,
                                                                                                 vis_keys_inter, self.T)

        # Combine losses
        combined_loss = (self.weight_intra_vision * vis_loss_intra
                         + self.weight_intra_tactile * tac_loss_intra
                         + self.weight_inter_tac_vis * vis_tac_inter
                         + self.weight_inter_vis_tac * tac_vis_inter)

        # # Perform momentum update during the forward pass
        # momentum_update_key_encoder(self.vision_base_q, self.vision_base_k, self.vision_head_intra_q,
        #                             self.vision_head_intra_k, self.m)
        # momentum_update_key_encoder(self.tactile_base_q, self.tactile_base_k, self.tactile_head_intra_q,
        #                             self.tactile_head_intra_k, self.m)
        logits = torch.cat([logits_vis_intra, logits_tact_intra, logits_vis_tac_inter, logits_tac_vis_inter], dim=0)
        labels = torch.cat(
            [labels_vis_intra, labels_tac_intra, labels_vision_tactile_inter, labels_tactile_vision_inter], dim=0)
        return combined_loss, logits, labels

    def log_losses(self, epoch, i, len_train_dataloader, vision_loss_intra, tactile_loss_intra, vision_tactile_inter,
                   tactile_vision_inter):
        """
        Log the computed losses using wandb.

        Parameters:
        - epoch: Current epoch number.
        - i: Current batch index.
        - len_train_dataloader: Total number of batches.
        - vision_loss_intra: Intra-modality contrastive loss for vision.
        - tactile_loss_intra: Intra-modality contrastive loss for tactile.
        - vision_tactile_inter: Inter-modality contrastive loss between vision and tactile.
        - tactile_vision_inter: Inter-modality contrastive loss between tactile and vision.
        """
        wandb.log({
            'module loss/vision intra loss': vision_loss_intra.item(),
            'module loss/tactile intra loss': tactile_loss_intra.item(),
            'module loss/vision tactile inter loss': vision_tactile_inter.item() * self.weight_inter,
            'module loss/tactile vision inter loss': tactile_vision_inter.item() * self.weight_inter
        }, step=epoch * len_train_dataloader + i)
