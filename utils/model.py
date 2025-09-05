import torch
import torch.nn as nn

class LUR(nn.Module):
    def __init__(self, 
                 input_dim,
                 num_classes,
                 act=None,
                 fc=None,
                 model=None,
                 num_projections=5):
        super(LUR, self).__init__()
        # Example split of the model and the final layer 
        self.model =nn.Sequential(*list(model.children())[:-1]).eval()
        self.fc = nn.Sequential(*list(model.children())[-1]).eval()    
        self.projections = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_projections)])
        self.act = act
    
    def forward(self, x):
        x = self.model(x)
        z = [x]
        y_hat = []
        # process the latent representation through the new layers
        for i, proj in enumerate(self.projections):
            z_p = proj(x)
            if self.act:
                z_p = self.act(z_p)
            z.append(z_p)
          
        for z_p in z:
            y_hat.append(self.fc(z_p))
        return z, y_hat
