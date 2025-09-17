## Latent Uncertainty Representations for Video-based Driver Action and Intention Recognition

### TLDR:
The introduction of trainable transformation layers and extending the prediction loss function improves the OOD detection performance.

### Abstract
Deep neural networks (DNNs) are increasingly applied to safety-critical tasks in resource-constrained environments, such as video-based driver action and intention recognition. While last layer probabilistic deep learning (LL--PDL) methods can detect out-of-distribution (OOD) instances, their performance varies. As an alternative to last layer approaches, we propose extending pre-trained DNNs with transformation layers to produce multiple latent representations to estimate the uncertainty. We evaluate our latent uncertainty representation (LUR) and repulsively trained LUR (RLUR) approaches against eight PDL methods across four video-based driver action and intention recognition datasets, comparing classification performance, calibration, and uncertainty-based OOD detection. We also contribute 28,000 frame-level action labels and 1,194 video-level intention labels for the NuScenes dataset. Our results show that LUR and RLUR achieve comparable in-distribution classification performance to other LL--PDL approaches. For uncertainty-based OOD detection, LUR matches top-performing PDL methods while being more efficient to train and easier to tune than approaches that require Markov-Chain Monte Carlo sampling or repulsive training procedures.



## Implementation
Integrating LUR into standard models is simple, and only requires adding the projection layers, updating the forward pass and the loss function to guide the training. 

**Original architecture:**
```python 
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.input = nn.Linear(in_features, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_features)
        self.activation = nn.ReLU()

    def forward(self, x, y=None):
        x = self.activation(self.input(x))
        x = self.activation(self.hidden(x))
        return self.fc(x)
```
**Modified architecture:**
```python 
class LUR_MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_dim,
                 out_features,
                 num_transformations):
        super().__init__()
        self.input = nn.Linear(in_features, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.trans_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_transformations)])
        self.fc = nn.Linear(hidden_dim, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input(x))
        z = self.activation(self.hidden(x))
        y = self.output(z)
        output = [y]

        for i, trans in enumerate(self.trans_layers):
            z_p = self.activation(trans(z))
            output.append(self.fc(z_p))
        return output
```

### Loss function example

```python
criterion = nn.CrossEntropyLoss()
output = model(x) # returns the prediction set
loss = torch.stack([criterion(y, target) for y in output]).sum()
```
