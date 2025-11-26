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
```python
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
```

2-qubit VQC with basis encoding, one CNOT entangling layer, and four trainable RY rotations.

(2) Truth Table of XOR
```python
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([0., 1., 1., 0.])
```
This is the complete XOR with four inputs. Y is the target output label.

(3) Parameter initialization + optimizer
```python
np.random.seed(0)
theta = np.random.uniform(0, 2*np.pi, 4)
theta = np.array(theta, requires_grad=True)

opt = qml.GradientDescentOptimizer(stepsize=0.4)
```
There are 4 trainable parameters for theta, which correspond to 4 RY gates. 
The step size is 0.4, this is the learning rate of gradient descent. 
The purpose of fixing the random seed is to ensure reproducibility.

(4) Loss function: MSE
```python
def cost(theta):
    loss = 0
    for x, y in zip(X, Y):
        probs = xor_circuit(x, theta)
        p1 = probs[1]
        loss += (p1 - y) ** 2
    return loss / len(X)
```
We run the quantum circuit for each of the four inputs one by one and take the output p(1). We use the mean squared error (MSE), which is a typical training approach for VQC regression/classification.

(5) Training loop + recording loss
```python
num_steps = 200
loss_history = []

for step in range(1, num_steps + 1):
    theta, loss = opt.step_and_cost(cost, theta)
    loss_history.append(loss)
```
We set step to 200, update theta at each step and record the loss. We printed the initial loss, several intermediate steps and the final loss.

(6) Analysis output after training completion
```python
print("Trained theta (rad):", theta)

for x, y in zip(X, Y):
    probs = xor_circuit(x, theta)
    p1 = probs[1]
    print(f"Input {x} | target = {y:.1f} | model p(1) = {p1:.4f}")
```
Print the final trained parameters θ (in radians and degrees). 
Output the predicted probabilities for the four inputs.
(7) Convergent Curve
```python
plt.figure(figsize=(6,4))
plt.plot(range(1, num_steps + 1), loss_history, linewidth=2)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training loss vs steps (XOR VQC)")
plt.grid(True)
plt.tight_layout()
plt.show()
```
Draw the curve of the loss decreasing with the number of steps during the training process.
The script reports the optimized parameters, final loss, prediction probabilities for all four XOR inputs, and plots the training loss curve, demonstrating that the VQC successfully learns the nonlinear XOR mapping.
