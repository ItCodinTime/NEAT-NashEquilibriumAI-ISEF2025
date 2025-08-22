#!/usr/bin/env python3
"""
NEAT Training Script with Iris Dataset
Train the NEAT model on the classic Iris classification task
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import neat_main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat_main import NEATAgent, NEATTrainer

class IrisNEATTrainer:
    """NEAT trainer specifically designed for Iris dataset"""
    
    def __init__(self, num_agents=3, hidden_dim=64, learning_rate=0.001):
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.load_data()
        
        # Initialize NEAT trainer
        self.trainer = NEATTrainer(
            num_agents=self.num_agents,
            input_dim=4,  # Iris has 4 features
            hidden_dim=self.hidden_dim,
            output_dim=3  # Iris has 3 classes
        )
        
        self.training_history = {
            'train_losses': [],
            'val_accuracies': [],
            'nash_equilibrium_epochs': [],
            'agent_losses': [[] for _ in range(num_agents)]
        }
    
    def load_data(self):
        """Load and preprocess Iris dataset"""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
    
    def validate(self):
        """Evaluate model on validation set"""
        # Use the first agent for validation (all should be similar at equilibrium)
        agent = self.trainer.agents[0]
        agent.eval()
        
        with torch.no_grad():
            outputs = agent(self.X_val)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == self.y_val).float().mean().item()
        
        agent.train()
        return accuracy
    
    def train(self, epochs=100, patience=10):
        """Train the NEAT model"""
        print("Starting NEAT training on Iris dataset...")
        print(f"Configuration: {self.num_agents} agents, {self.hidden_dim} hidden units")
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_x, batch_y in self.train_loader:
                losses = self.trainer.train_step(batch_x, batch_y)
                epoch_losses.append(losses)
                
                # Store individual agent losses
                for i, loss in enumerate(losses):
                    self.training_history['agent_losses'][i].append(loss)
            
            # Calculate average losses
            avg_losses = np.mean(epoch_losses, axis=0)
            avg_total_loss = np.mean(avg_losses)
            
            # Validate
            val_accuracy = self.validate()
            
            # Store history
            self.training_history['train_losses'].append(avg_total_loss)
            self.training_history['val_accuracies'].append(val_accuracy)
            
            # Check for Nash equilibrium
            if self.trainer.is_nash_equilibrium(self.X_val, self.y_val):
                self.training_history['nash_equilibrium_epochs'].append(epoch)
                print(f"Nash equilibrium detected at epoch {epoch}!")
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter == 0:
                print(f"Epoch {epoch:3d}: Loss={avg_total_loss:.4f}, Val_Acc={val_accuracy:.4f}")
                print(f"  Agent losses: {[f'{l:.4f}' for l in avg_losses]}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break
        
        print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
        return self.training_history
    
    def test(self):
        """Evaluate on test set"""
        # Load best model
        if os.path.exists('models/best_model.pth'):
            self.trainer.agents[0].load_state_dict(torch.load('models/best_model.pth'))
        
        agent = self.trainer.agents[0]
        agent.eval()
        
        with torch.no_grad():
            outputs = agent(self.X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(self.y_test.numpy(), predicted.numpy())
            
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test.numpy(), predicted.numpy(),
                                    target_names=['Setosa', 'Versicolor', 'Virginica']))
        
        return accuracy, predicted.numpy()
    
    def save_model(self, filename):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        torch.save(self.trainer.agents[0].state_dict(), f'models/{filename}')
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_history['train_losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Validation accuracy
        axes[0, 1].plot(self.training_history['val_accuracies'])
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # Individual agent losses
        for i, agent_losses in enumerate(self.training_history['agent_losses']):
            axes[1, 0].plot(agent_losses, label=f'Agent {i+1}')
        axes[1, 0].set_title('Individual Agent Losses')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Nash equilibrium epochs
        if self.training_history['nash_equilibrium_epochs']:
            axes[1, 1].bar(range(len(self.training_history['nash_equilibrium_epochs'])),
                          self.training_history['nash_equilibrium_epochs'])
            axes[1, 1].set_title('Nash Equilibrium Detection')
            axes[1, 1].set_xlabel('Detection Instance')
            axes[1, 1].set_ylabel('Epoch')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Nash Equilibrium\nDetected', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Nash Equilibrium Detection')
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, test_accuracy, predictions):
        """Save training results to JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_agents': self.num_agents,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate
            },
            'training_history': self.training_history,
            'test_accuracy': test_accuracy,
            'nash_equilibrium_count': len(self.training_history['nash_equilibrium_epochs']),
            'final_validation_accuracy': self.training_history['val_accuracies'][-1] if self.training_history['val_accuracies'] else 0
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/iris_neat_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to results/iris_neat_results.json")

def main():
    """Main training function"""
    print("NEAT Training on Iris Dataset")
    print("=" * 40)
    
    # Initialize trainer
    trainer = IrisNEATTrainer(
        num_agents=3,
        hidden_dim=64,
        learning_rate=0.001
    )
    
    # Train model
    history = trainer.train(epochs=200, patience=20)
    
    # Test model
    test_accuracy, predictions = trainer.test()
    
    # Plot results
    trainer.plot_training_history()
    
    # Save results
    trainer.save_results(test_accuracy, predictions)
    
    # Final model save
    trainer.save_model('neat_iris_final.pth')
    
    print("\nTraining completed successfully!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Nash equilibrium achieved {len(history['nash_equilibrium_epochs'])} times")

if __name__ == "__main__":
    main()
