# RBF-fPINN: 
## Fractional Physics-Informed Neural Network with Radial Basis Functions

Our package introduces **RBF-fPINN**, a cutting-edge *fractional Physics-Informed Neural Network (fPINN)* that leverages **Radial Basis Function (RBF)** neural networks to significantly enhance solution accuracy. Unlike traditional fPINNs, the **RBF-fPINN** employs a *single hidden layer* where **RBFs** serve as activation functions. We perform
all simulations with the PyTorch framework on a single NVIDIA Tesla V100 GPU.

For more information, please refer to our paper:

Mohammadi, M., Mokhtari, R., & Ramezani, M. Rbf-Fpinns: Radial Basis Function-Enhanced Fractional Physics-Informed Neural Networks. Under Reviewing. (Available at SSRN 5072679)

---
## Running Example 1

- By default, **RBF-fPINN** results are shown.  
- To view **fPINN** results instead, change the following line in the code:

```python
use_RBFfPINN = True  # Set to False to run standard fPINN
```
---

## ⚖️ License

This project is released under the **MIT License**.  

> © 2025 — Please cite the corresponding paper if you use this repository in your research.

---

## Citation

If you find this useful in your research, please cite it as:

```bibtex
@article{ramazani2025rbf-fpinn,
  title     = {RBF-fPINN: A Radial Basis Function Enhanced Fractional Physics-Informed Neural Network},
  author    = {Maryam Mohammadi, Reza Mokhtari, Mohadese Ramezani},
  journal   = {UnderReview},
  year      = {2025}
}

