# 🧠 RBF-fPINN: Fractional Physics-Informed Neural Network with Radial Basis Functions

This repository contains the implementation of the paper:

> **RBF-fPINN: A Radial Basis Function Enhanced Fractional Physics-Informed Neural Network**

---

## 📘 Overview

This paper introduces **RBF-fPINN**, a cutting-edge *fractional Physics-Informed Neural Network (fPINN)* that leverages **Radial Basis Function (RBF)** neural networks to significantly enhance solution accuracy.  

Unlike traditional fPINNs, the **RBF-fPINN** employs a *single hidden layer* where **RBFs** serve as activation functions. Three widely used radial basis functions are evaluated:  
- Gaussian  
- Inverse Quadratic  
- Inverse Multiquadric  

The study identifies the most effective RBF formulation for solving **fractional partial differential equations (PDEs)**.

These equations, characterized by **fractional derivatives**, capture *non-local dynamics* and *anomalous transport processes* in diverse engineering and physical systems. Traditional numerical methods often struggle with these problems, particularly in **high-dimensional** or **irregular domains**.

Our proposed **RBF-fPINN** framework demonstrates remarkable stability and accuracy through extensive numerical experiments — even for complex **2D** and **3D** geometries.  
It shows strong potential for improving the performance of neural PDE solvers in **fractional and mobile–immobile (MIM)** models.

---

## ✨ Key Features

- Integration of **RBFs** into PINN structure for improved accuracy.  
- Handles **fractional derivatives** and **non-local operators** effectively.  
- Supports complex and **irregular 2D / 3D geometries**.  
- Robust performance in **MIM (mobile–immobile)** equations.  
- Easily extensible for **multi-dimensional** fractional PDEs.  

---

## 👩‍🔬 Authors

- **Maryam Mohammadi** — Ph.D. Student, Department of Mathematical Sciences, Isfahan University of Technology, Iran.  
- **Reza Mokhtari** — Professor, Department of Applied Mathematics, Isfahan University of Technology, Iran.  
- **Mahdieh Ramazani** — Postdoctoral Researcher, Department of Mathematical Sciences, Isfahan University of Technology, Iran.

---

## ⚖️ License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute the code, provided that proper credit is given to the original authors.

> © 2025 — Please cite the corresponding paper if you use this repository in your research.

---

## 📄 Citation

If you use this work in your research, please cite it as:

```bibtex
@article{ramazani2025rbf-fpinn,
  title     = {RBF-fPINN: A Radial Basis Function Enhanced Fractional Physics-Informed Neural Network},
  author    = {Maryam Mohammadi, Reza Mokhtari, Mohadese Ramezani},
  journal   = {UnderReview},
  year      = {2025}
}

