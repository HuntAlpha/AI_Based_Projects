!pip install transformers pillow faiss-cpu numpy torch gradio --quiet
!pip install opendatasets --quiet

import os
import numpy as np
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel
import faiss
import torch
from tqdm import tqdm
import gradio as gr
import opendatasets as od
import shutil

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ReadyToUseVisualSearch:
    def __init__(self):
        # Automatic device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Initializing on {self.device.upper()} device")

        # Load model with error handling
        try:
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                device_map="auto"
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

        self.image_paths = []
        self.index = None

    def download_sample_dataset(self):
        """Download and prepare the Flickr8k dataset"""
        print("⬇️ Downloading sample dataset...")
        try:
            od.download("https://www.kaggle.com/datasets/adityajn105/flickr8k")
            # Create clean image directory
            os.makedirs("flickr_images", exist_ok=True)
            # Move images to clean folder
            for img in os.listdir("flickr8k/Images"):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.move(f"flickr8k/Images/{img}", "flickr_images/")
            return "flickr_images"
        except Exception as e:
            print(f"❌ Dataset download failed: {e}")
            # Fallback to placeholder images
            self.create_fallback_images()
            return "fallback_images"

    def create_fallback_images(self):
        """Create simple fallback images if download fails"""
        os.makedirs("fallback_images", exist_ok=True)
        !wget -q -O fallback_images/cat.jpg https://placekitten.com/400/300
        !wget -q -O fallback_images/dog.jpg https://placedog.net/400/300
        !wget -q -O fallback_images/beach.jpg https://source.unsplash.com/random/400x300/?beach

    def build_index(self, image_folder, max_images=200):
        """Safe index builder with progress tracking"""
        print("🔍 Preparing images...")

        # Get image paths with safety limit
        self.image_paths = []
        for file in os.listdir(image_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(image_folder, file))
                if len(self.image_paths) >= max_images:
                    break

        print(f"🖼️ Processing {len(self.image_paths)} images...")

        # Process images with error handling
        embeddings = []
        valid_paths = []

        for path in tqdm(self.image_paths, desc="📊 Generating embeddings"):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")  # Ensure RGB format
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        embedding = self.model.get_image_features(**inputs).cpu().numpy()
                    embeddings.append(embedding)
                    valid_paths.append(path)
            except Exception as e:
                print(f"⚠️ Skipped {os.path.basename(path)}: {str(e)}")
                continue

        if not embeddings:
            raise ValueError("❌ No valid images processed!")

        # Build FAISS index
        print("🧩 Creating search index...")
        self.image_paths = valid_paths
        embeddings = np.vstack(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"🎉 Index ready with {len(self.image_paths)} images!")

    def search(self, query, top_k=5):
        """Robust search with multiple fallbacks"""
        try:
            # Handle different query types
            if isinstance(query, str):
                if query.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        with Image.open(query) as img:
                            query_embed = self.get_embedding(image=img)
                    except:
                        return []
                else:
                    query_embed = self.get_embedding(text=query)
            else:  # PIL Image
                query_embed = self.get_embedding(image=query)

            if query_embed is None:
                return []

            # Normalize and search
            query_embed = query_embed.astype('float32')
            faiss.normalize_L2(query_embed)
            distances, indices = self.index.search(query_embed, top_k)

            # Return only valid, accessible images
            results = []
            for i, score in zip(indices[0], distances[0]):
                if 0 <= i < len(self.image_paths):
                    img_path = self.image_paths[i]
                    try:
                        with Image.open(img_path) as _:  # Test if image can be opened
                            results.append(img_path)
                    except:
                        continue
            return results[:top_k]

        except Exception as e:
            print(f"🔴 Search error: {e}")
            return []

    def get_embedding(self, image=None, text=None):
        """Safe embedding generator"""
        try:
            with torch.no_grad():
                if image is not None:
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    return self.model.get_image_features(**inputs).cpu().numpy()
                elif text is not None:
                    inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                    return self.model.get_text_features(**inputs).cpu().numpy()
        except Exception as e:
            print(f"🔴 Embedding error: {e}")
            return None

def create_interface(engine):
    def search_interface(query):
        try:
            results = engine.search(query)
            return results if results else ["No results found"]
        except:
            return ["Search error occurred"]

    return gr.Interface(
        fn=search_interface,
        inputs=gr.Textbox(label="Search by text or image path"),
        outputs=gr.Gallery(
            label="Results",
            columns=3,
            height="auto",
            object_fit="contain"
        ),
        examples=["cat", "dog", "beach", "person smiling"],
        title="🔍 Visual Search Engine",
        description="Search for images using text queries or example images"
    )

def main():
    # Initialize the engine
    engine = ReadyToUseVisualSearch()

    # Download and prepare dataset
    image_folder = engine.download_sample_dataset()

    # Build index
    engine.build_index(image_folder)

    # Create and launch interface
    interface = create_interface(engine)
    interface.launch(share=True)  # Set share=False if you don't want a public link

if __name__ == "__main__":
    print("🚀 Starting visual search engine...")
    main()