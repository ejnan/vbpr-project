import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=8, dropout=0, mean=0):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_visual_emb = nn.Embedding(num_users, visual_projection_size)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)
        self.dropout = nn.Dropout(dropout)

        self.num_user = num_users

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return self.dropout((U * I).sum(1) + b_u + b_i + self.mean)
    
class VBPR(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64, visual_size=512, category_size=16, visual_projection_size=64, dropout=0, mean=0):

        super(VBPR, self).__init__()
        
        # Following the architecture from the lecture:
        # Pretrained deep CNN -> visual features (4096 x 1) -> Embedding -> 
        # Item Visual Factors (D x 1) and Item Latent Factors (F x 1) -> 
        # Item Factors and User Factors -> Prediction taking into account Biases
        
        # User components
        # Traditional user latent factors for CF
        self.user_emb = nn.Embedding(num_users, embedding_size)
        # User visual preference factors
        self.user_visual_emb = nn.Embedding(num_users, visual_projection_size)
        # User category preference factors
        self.user_category_emb = nn.Embedding(num_users, embedding_size)
        # User bias term
        self.user_bias = nn.Embedding(num_users, 1)
        
        # Item components
        # Traditional item latent factors for CF
        self.item_emb = nn.Embedding(num_items, embedding_size)
        # Item bias term
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Visual features projection (from high-dimensional CNN features to D dimensions)
        self.visual_projection = nn.Linear(visual_size, visual_projection_size)
        
        # Category features
        self.max_category_id = 0
        self.category_size = category_size
        self.category_emb = None
        self.item_category_projection = nn.Linear(category_size, embedding_size)
        
        # Initialize weights
        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        self.user_visual_emb.weight.data.uniform_(0, 0.005)
        self.user_category_emb.weight.data.uniform_(0, 0.005)
        self.user_category_emb.weight.data.uniform_(0, 0.005)
        
        # Global bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)
        self.dropout = nn.Dropout(dropout)
        
        # Store dimensions for reference
        self.num_users = num_users
        self.visual_size = visual_size
        self.visual_projection_size = visual_projection_size
        
        # Features to be loaded externally
        self.visual_features = None
        self.category_features = None
        
    def set_visual_features(self, visual_features):

        self.visual_features = visual_features
        
    def set_category_features(self, category_features):
        
        self.category_features = category_features
        
        # Determine max category ID to properly size the embedding table
        if category_features:
            max_id = max(category_features.values()) + 1  # +1 because embeddings are 0-indexed
            self.max_category_id = max(self.max_category_id, max_id)

            print(f"Initializing category embedding with size: {self.max_category_id} x {self.category_size}")
            
            # Initialize category embedding now that we know the size
            self.category_emb = nn.Embedding(self.max_category_id, self.category_size)
            self.category_emb.weight.data.uniform_(0, 0.005)
                
            # If model is on a device, move embedding to the same device
            device = self.user_emb.weight.device
            if device.type != 'cpu':
                self.category_emb = self.category_emb.to(device)
        
    def forward(self, u_id, i_id):
        """
        Forward pass for prediction.
        
        Args:
            u_id: User IDs
            i_id: Item IDs
        
        Returns:
            Predicted preference scores
        """
        device = u_id.device
        
        # Get user and item embeddings (CF component)
        U = self.user_emb(u_id)  # User latent factors for CF
        b_u = self.user_bias(u_id).squeeze()  # User bias
        I = self.item_emb(i_id)  # Item latent factors for CF
        b_i = self.item_bias(i_id).squeeze()  # Item bias
        
        # Get user visual preferences
        U_v = self.user_visual_emb(u_id)  # User visual preference factors
        
        # Calculate traditional CF component: <U, I>
        mf_pred = torch.sum(U * I, dim=1)
        
        # Visual component: <U_v, E_v>
        visual_pred = torch.zeros(len(u_id), device=device)
        if self.visual_features is not None:
            # Process visual features
            visual_features = torch.zeros((len(i_id), self.visual_size), device=device)
            
            for idx, item_id in enumerate(i_id.cpu().numpy()):
                if item_id in self.visual_features:
                    visual_features[idx] = torch.FloatTensor(self.visual_features[item_id]).to(device)
            
            # Project visual features through embedding layer: CNN features -> Item Visual Factors
            I_v = self.visual_projection(visual_features)
            
            # Calculate visual component prediction: <U_v, I_v>
            visual_pred = torch.sum(U_v * I_v, dim=1)
        
        # Category component (extension to original VBPR)
        category_pred = torch.zeros(len(u_id), device=device)
        if self.category_features is not None and self.category_emb is not None:
            # Get user category preferences
            U_c = self.user_category_emb(u_id)
            
            # Process category features
            category_ids = torch.zeros(len(i_id), dtype=torch.long, device=device)
            
            for idx, item_id in enumerate(i_id.cpu().numpy()):
                if item_id in self.category_features:
                    cat_id = self.category_features[item_id]

                    if cat_id < self.max_category_id:
                        category_ids[idx] = cat_id
            
            # Get category embeddings
            C_emb = self.category_emb(category_ids)
            
            # Project category embeddings to the same space as user embeddings
            I_c = self.item_category_projection(C_emb)
            
            # Calculate category component prediction
            category_pred = torch.sum(U_c * I_c, dim=1)
        
        # Combine all components for final prediction: 
        # CF component + Visual component + Category component + Biases
        prediction = mf_pred + visual_pred + category_pred + b_u + b_i + self.mean
        
        return self.dropout(prediction)