import os
import warnings
import torch
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    chunk_ids: List[str]

class EmbeddingEngine:
    def __init__(self, model_name: str = "intfloat/e5-small-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.embedding_dim = 384

    def load_model(self):
        """Load embedding model with bulletproof offline enforcement"""
        print(f"Loading embedding model: {self.model_name}")
        
        # Set comprehensive offline environment
        offline_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", 
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_DATASETS_OFFLINE": "1",
            "DISABLE_HUGGINGFACE_HUB_CACHE_CHECK": "1"
        }
        
        for key, value in offline_env.items():
            os.environ[key] = value
        
        # Try SentenceTransformer first (preferred method)
        success_method = None
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Method 1: SentenceTransformer with enhanced offline parameters
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                use_auth_token=False,
                trust_remote_code=False,
                cache_folder=None
            )
            
            self.model.eval()
            success_method = "SentenceTransformer"
            print(f"✅ Model loaded successfully on {self.device} via {success_method}")
            
        except Exception as e1:
            print(f"⚠️ SentenceTransformer failed: {str(e1)[:100]}...")
            
            # Method 2: Direct transformers library approach
            try:
                from transformers import AutoModel, AutoTokenizer
                
                print("🔄 Trying direct transformers approach...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    trust_remote_code=False
                )
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    trust_remote_code=False
                )
                
                self.model = self.model.to(self.device)
                self.model.eval()
                success_method = "Direct Transformers"
                print(f"✅ Model loaded successfully on {self.device} via {success_method}")
                
            except Exception as e2:
                print(f"❌ Direct transformers also failed: {str(e2)[:100]}...")
                
                # Method 3: Alternative cache path attempt
                try:
                    print("🔄 Trying alternative cache approach...")
                    
                    # Get user's home directory cache
                    home_dir = os.path.expanduser("~")
                    cache_paths = [
                        os.path.join(home_dir, ".cache", "huggingface", "hub"),
                        os.path.join(home_dir, ".cache", "torch", "sentence_transformers"),
                        os.path.join(home_dir, ".cache", "huggingface", "transformers")
                    ]
                    
                    for cache_path in cache_paths:
                        if os.path.exists(cache_path):
                            print(f"🔍 Found cache directory: {cache_path}")
                    
                    # Try with explicit cache folder
                    from sentence_transformers import SentenceTransformer
                    
                    self.model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=cache_paths[0] if os.path.exists(cache_paths[0]) else None
                    )
                    
                    success_method = "Alternative Cache"
                    print(f"✅ Model loaded successfully on {self.device} via {success_method}")
                    
                except Exception as e3:
                    print(f"❌ All loading methods failed.")
                    print(f"Final error: {str(e3)}")
                    print("\n🔧 Troubleshooting suggestions:")
                    print("1. Check if models are actually cached:")
                    print(f"   ls ~/.cache/huggingface/")
                    print("2. Try clearing cache and re-downloading:")
                    print(f"   rm -rf ~/.cache/huggingface/")
                    print(f"   python complete_model_download.py")
                    raise Exception("Failed to load embedding model with all methods")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if self.model is None:
            self.load_model()
        
        # Check if we're using SentenceTransformer or direct transformers
        if hasattr(self.model, 'encode'):
            # SentenceTransformer method
            query_with_prefix = f"query: {query}"
            embedding = self.model.encode(query_with_prefix, convert_to_tensor=True)
            return embedding.cpu().numpy().flatten()
        else:
            # Direct transformers method
            query_with_prefix = f"query: {query}"
            
            inputs = self.tokenizer(
                query_with_prefix, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.cpu().numpy().flatten()
    
    def embed_chunks(self, chunks: List[Any]) -> EmbeddingResult:
        """Generate embeddings for multiple chunks"""
        if self.model is None:
            self.load_model()
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = []
        chunk_ids = []
        
        # Process in batches to manage memory
        batch_size = 32
        from tqdm import tqdm
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = []
            batch_chunk_ids = []
            
            for chunk in batch_chunks:
                # Add prefix for E5 model
                text = f"passage: {chunk.content}"
                batch_texts.append(text)
                batch_chunk_ids.append(chunk.chunk_id)
            
            # Generate embeddings for batch
            if hasattr(self.model, 'encode'):
                # SentenceTransformer batch encoding
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                batch_embeddings = batch_embeddings.cpu().numpy()
            else:
                # Direct transformers batch encoding
                batch_embeddings = []
                for text in batch_texts:
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1)
                        batch_embeddings.append(embedding.cpu().numpy().flatten())
                
                batch_embeddings = np.vstack(batch_embeddings)
            
            embeddings.extend(batch_embeddings)
            chunk_ids.extend(batch_chunk_ids)
        
        final_embeddings = np.vstack(embeddings) if len(embeddings) > 1 else np.array(embeddings)
        
        print(f"✅ Generated embeddings shape: {final_embeddings.shape}")
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            chunk_ids=chunk_ids
        )
