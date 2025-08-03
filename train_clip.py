import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
import logging
from typing import List, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MaterialDataset(Dataset):
    def __init__(self, image_paths: List[str], captions: List[str], processor: CLIPProcessor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Process image and text
        inputs = self.processor(
            images=image,
            text=self.captions[idx],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Remove batch dimension
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].squeeze(0)
        
        return inputs

class CLIPTrainer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        device: str = None
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize wandb for tracking
        wandb.init(project="material-clip-training")
    
    def prepare_data(
        self,
        data_dir: str,
        annotations_file: str
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders"""
        # Load annotations
        df = pd.read_csv(annotations_file)
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        
        # Create datasets
        train_dataset = MaterialDataset(
            image_paths=[os.path.join(data_dir, path) for path in train_df['image_path']],
            captions=train_df['caption'].tolist(),
            processor=self.processor
        )
        
        val_dataset = MaterialDataset(
            image_paths=[os.path.join(data_dir, path) for path in val_df['image_path']],
            captions=val_df['caption'].tolist(),
            processor=self.processor
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str
    ):
        """Train the CLIP model"""
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Log metrics
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(os.path.join(save_dir, 'best_model'))
            
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    def save_model(self, path: str):
        """Save the fine-tuned model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
    
    def load_model(self, path: str):
        """Load a fine-tuned model"""
        self.model = CLIPModel.from_pretrained(path)
        self.processor = CLIPProcessor.from_pretrained(path)
        self.model.to(self.device)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = CLIPTrainer(
        batch_size=32,
        learning_rate=2e-5,
        num_epochs=10
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        data_dir='data/pinterest_materials',
        annotations_file='data/pinterest_annotations.csv'
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir='models/clip_material'
    )

if __name__ == "__main__":
    main() 