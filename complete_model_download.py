import os
import torch
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

def force_download_all_models():
    """Forcefully download and verify all required models"""
    print("🔄 Force downloading all models with verification...")
    
    # Set cache directory explicitly
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"📁 Cache directory: {cache_dir}")
    
    models_to_download = [
        {
            'name': 'intfloat/e5-small-v2',
            'type': 'embedding',
            'loader': SentenceTransformer
        },
        {
            'name': 'google/flan-t5-base', 
            'type': 't5',
            'loader_model': T5ForConditionalGeneration,
            'loader_tokenizer': T5Tokenizer
        },
        {
            'name': 'distilbert-base-uncased-distilled-squad',
            'type': 'qa',
            'loader_model': AutoModelForQuestionAnswering,
            'loader_tokenizer': AutoTokenizer
        }
    ]
    
    for model_info in models_to_download:
        try:
            print(f"\n📥 Downloading {model_info['name']}...")
            
            if model_info['type'] == 'embedding':
                # Download SentenceTransformer model
                model = model_info['loader'](model_info['name'])
                print(f"✅ {model_info['name']} downloaded successfully")
                
            else:
                # Download regular transformers models
                tokenizer = model_info['loader_tokenizer'].from_pretrained(
                    model_info['name'],
                    cache_dir=cache_dir,
                    force_download=False  # Don't re-download if exists
                )
                
                model = model_info['loader_model'].from_pretrained(
                    model_info['name'],
                    cache_dir=cache_dir,
                    force_download=False
                )
                
                print(f"✅ {model_info['name']} downloaded successfully")
                
            # Verify download by loading in offline mode
            print(f"🔍 Verifying {model_info['name']} offline accessibility...")
            
            if model_info['type'] == 'embedding':
                # Test offline loading for SentenceTransformer
                os.environ["HF_HUB_OFFLINE"] = "1"
                test_model = SentenceTransformer(model_info['name'])
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                # Test offline loading for regular models
                test_tokenizer = model_info['loader_tokenizer'].from_pretrained(
                    model_info['name'],
                    local_files_only=True
                )
                test_model = model_info['loader_model'].from_pretrained(
                    model_info['name'], 
                    local_files_only=True
                )
            
            print(f"✅ {model_info['name']} verified for offline use")
            
        except Exception as e:
            print(f"❌ Error with {model_info['name']}: {str(e)}")
            return False
    
    print("\n🎉 All models downloaded and verified for offline operation!")
    print("📊 Your RAG chatbot is now ready!")
    return True

if __name__ == "__main__":
    success = force_download_all_models()
    if not success:
        print("\n❌ Model download failed. Check your internet connection and try again.")
    else:
        print("\n✅ All models ready! You can now run your RAG chatbot offline.")
