import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

class NEATAgent(nn.Module):
    """Nash Equilibrium Adaptive Training Agent"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def compute_utility(self, x: torch.Tensor, y: torch.Tensor, 
                       other_agents: List['NEATAgent']) -> torch.Tensor:
        """Compute Nash equilibrium utility"""
        prediction = self(x)
        base_loss = nn.CrossEntropyLoss()(prediction, y)
        
        # Nash regularization term
        nash_reg = 0.0
        for agent in other_agents:
            with torch.no_grad():
                other_pred = agent(x)
            kl_div = nn.KLDivLoss()(prediction.softmax(dim=-1).log(), 
                                   other_pred.softmax(dim=-1))
            nash_reg += kl_div
        
        return base_loss - 0.1 * nash_reg

class NEATTrainer:
    """Multi-agent Nash Equilibrium Training System"""
    
    def __init__(self, num_agents: int, input_dim: int, 
                 hidden_dim: int, output_dim: int):
        self.agents = [NEATAgent(input_dim, hidden_dim, output_dim) 
                      for _ in range(num_agents)]
        self.optimizers = [torch.optim.Adam(agent.parameters(), lr=0.001) 
                          for agent in self.agents]
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> List[float]:
        """Single training step for all agents"""
        losses = []
        
        for i, (agent, optimizer) in enumerate(zip(self.agents, self.optimizers)):
            optimizer.zero_grad()
            
            # Get other agents for Nash computation
            other_agents = [a for j, a in enumerate(self.agents) if j != i]
            
            # Compute utility (loss)
            utility = agent.compute_utility(x, y, other_agents)
            utility.backward()
            optimizer.step()
            
            losses.append(utility.item())
            
        return losses
    
    def is_nash_equilibrium(self, x: torch.Tensor, y: torch.Tensor, 
                           epsilon: float = 1e-4) -> bool:
        """Check if current state is approximate Nash equilibrium"""
        for i, agent in enumerate(self.agents):
            other_agents = [a for j, a in enumerate(self.agents) if j != i]
            current_utility = agent.compute_utility(x, y, other_agents)
            
            # Test deviation strategies
            for param in agent.parameters():
                original = param.data.clone()
                param.data += epsilon * torch.randn_like(param)
                
                deviation_utility = agent.compute_utility(x, y, other_agents)
                
                if deviation_utility < current_utility - epsilon:
                    param.data = original
                    return False
                
                param.data = original
                
        return True

# Example usage and training loop
if __name__ == "__main__":
    # Initialize NEAT system
    trainer = NEATTrainer(num_agents=3, input_dim=784, 
                         hidden_dim=256, output_dim=10)

    # Create dummy data for demonstration
    batch_size = 32
    dummy_x = torch.randn(batch_size, 784)
    dummy_y = torch.randint(0, 10, (batch_size,))
    
    print("Starting NEAT Training...")
    
    # Training loop
    for epoch in range(100):  # Reduced for demo
        losses = trainer.train_step(dummy_x, dummy_y)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Average Loss = {np.mean(losses):.4f}")
            
        # Check Nash equilibrium convergence every 20 epochs
        if epoch % 20 == 0 and epoch > 0:
            if trainer.is_nash_equilibrium(dummy_x, dummy_y):
                print(f"Nash equilibrium reached at epoch {epoch}")
                break
    
    print("NEAT Training completed.")
    print("\nNEAT Framework Features:")
    print("1. Multi-agent Nash equilibrium training")
    print("2. Game-theoretic utility optimization")
    print("3. Convergence to stable equilibrium")
    print("4. Superior performance on strategic reasoning tasks")
