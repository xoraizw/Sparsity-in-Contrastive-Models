# Sparsity in Contrastive Models

This repository explores the use of sparse signal recovery methods, such as \( l_1 \)-relaxation, to improve the interpretability of multimodal representations in contrastive models like CLIP (Contrastive Language–Image Pretraining). The primary objective is to evaluate how sparse CLIP embeddings compare to dense embeddings in a zero-shot classification task.

## Features

1. **Zero-Shot Classification with Dense Embeddings**
   - Use a pre-trained CLIP model to generate dense image and text embeddings.
   - Perform zero-shot classification by computing cosine similarity between embeddings.

2. **Inducing Sparsity in CLIP Embeddings**
   - Build a concept dictionary from human-readable tokens using CLIP's text encoder.
   - Enforce sparsity through \( l_1 \)-regularized optimization.
   - Align image and concept embeddings to a shared latent space.

3. **Sparse vs Dense Embeddings**
   - Compare classification accuracy, memory footprint, and computational advantages.
   - Analyze the relationship between sparsity level and accuracy.
   - Identify top contributing concepts and provide qualitative examples.

## Usage

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-username/sparsity-contrastive-models.git
cd sparsity-contrastive-models
pip install -r requirements.txt


### Data Preparation
We use the CIFAR-10 dataset for the experiments. The dataset will be automatically downloaded when running the scripts.

### Running the Project
1. **Generate Dense Embeddings**
   - Use the pre-trained CLIP model to extract dense image embeddings and text embeddings.

2. **Build Concept Dictionary**
   - Create a concept dictionary from CIFAR-10 classes or a larger vocabulary set.
   - Use CLIP's text encoder to embed concepts into a shared latent space.

3. **Sparse Embedding Generation**
   - Solve the optimization problem:
     \[
     \min_w ||z_c - A_c w||^2_2 + \lambda ||w||_1
     \]
   - Reconstruct sparse embeddings using the concept dictionary.

4. **Zero-Shot Classification**
   - Compute cosine similarity between embeddings for classification.
   - Compare results for dense and sparse embeddings.

5. **Evaluation**
   - Report accuracy, sparsity levels, memory footprint, and qualitative insights.

## Results
- **Accuracy**: Compare the zero-shot classification accuracy of dense vs sparse embeddings.
- **Sparsity-Accuracy Tradeoff**: Plot the relationship between sparsity and classification accuracy.
- **Memory Efficiency**: Highlight the memory savings of sparse embeddings.
- **Concept Analysis**: Identify which dictionary concepts contribute most to classification.

## Dependencies
- Python 3.10+
- PyTorch
- torchvision
- transformers (Hugging Face)
- timm
- matplotlib

## Example Code

### Loading Libraries and Dataset
```python
from models.model import CLIPModel
from models.utils import train_step, eval_step, DataLoaders
from torchvision.datasets import CIFAR10
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
train = CIFAR10(root='data', train=True, download=True, transform=transform)
test = CIFAR10(root='data', train=False, download=True, transform=transform)
```

### Building Concept Dictionary
```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
concept_dictionary = []

for caption in classes:
    encoded_captions = tokenizer(caption, padding=True, truncation=True, return_tensors='pt')
    text_emb = model.text_encoder(input_ids=encoded_captions['input_ids'], attention_mask=encoded_captions['attention_mask'])
    concept_dictionary.append(text_emb)
```

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The CLIP model is provided by OpenAI.
- CIFAR-10 dataset is sourced from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
```
