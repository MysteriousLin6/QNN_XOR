# Quantum Neural Network (QNN) Realization of XOR Using Variational Quantum Circuit (VQC)

This repository contains the full implementation of a Variational Quantum Circuit (VQC)–based Quantum Neural Network (QNN) for learning the XOR function.  
The experiment includes both simulation (PennyLane) and real hardware execution (SpinQ Desktop Quantum Computer).

---

## 📌 Project Overview

This study demonstrates how a 2-qubit VQC can learn a nonlinear Boolean function (XOR) using:

- **Angle embedding / basis encoding**
- **Entangling gates (CNOT)**
- **Trainable parameters (RY rotations)**
- **Gradient Descent Optimization**
- **Evaluation on real quantum hardware**

The trained VQC achieves:

- **Simulation accuracy:** 97%  
- **Experimental fidelity (SpinQ):** ≈98.85%  

This validates the feasibility of performing quantum machine learning on small-scale NISQ devices.

---

## 📁 Contents

