# Quantum Neural Network (QNN) Realization of XOR Using Variational Quantum Circuit (VQC)

This repository contains the full implementation of a Variational Quantum Circuit (VQC)–based Quantum Neural Network (QNN) for learning the XOR function. This code is designed to serve the purpose of running on the real hardware.  

## Project Overview

This study demonstrates how a 2-qubit VQC can learn a nonlinear Boolean function (XOR) using:

- **Angle embedding / basis encoding**
- **Entangling gates (CNOT)**
- **Trainable parameters (RY rotations)**
- **Gradient Descent Optimization**
- **Evaluation on real quantum hardware**


# Code structure
(1) Quantum devices + circuit definition

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def xor_circuit(x, theta):
    if x[0] == 1:
        qml.PauliX(0)
    if x[1] == 1:
        qml.PauliX(1)
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[2], wires=0)
    qml.RY(theta[3], wires=1)
    
    return qml.probs(wires=1)

2-qubit VQC with basis encoding, one CNOT entangling layer, and four trainable RY rotations.

(2) Truth Table of XOR

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([0., 1., 1., 0.])



