# Actorob

**Actorob** is a research repository accompanying the IEEE RA-L paper:

**[Task-Aware Actuator Parameter Allocation for Multibody Robots](https://ieeexplore.ieee.org/document/11433790)**

---

## Overview

Legged robots require actuator parameter sets that reconcile conflicting objectives such as high torque density, low reflected impedance, responsiveness, and energy efficiency. Traditional actuator sizing is often decoupled from motion generation, which can yield suboptimal system-level performance.

This repository implements a **task-aware co-design framework** that jointly optimizes actuator parameters and full-body motion for multibody robotic systems. The method is formulated as a nested optimization problem with two tightly coupled cycles:

### Outer loop (actuator allocation + parameter recovery)

The outer loop searches a continuous actuator design space (e.g., **motor mass** and **gear ratio** per joint) using **CMA-ES**. For every candidate design, actuator properties are **recovered in a data-driven manner** via regressions fitted to manufacturer catalogs, yielding:
- peak torque
- rotor inertia
- motor constant
- geometric scaling / motor dimensions
- stage-dependent gearbox efficiency

This loop produces task-specific actuator candidates grounded in realistic electromechanical characteristics.

### Inner loop (full-body constrained trajectory optimization)

For each actuator candidate produced by the outer loop, the inner loop solves a full-body trajectory optimization problem subject to:
- rigid-body dynamics constraints
- kinematic constraints
- contact consistency constraints
- actuator feasibility constraints (torque–velocity limits, max torque/speed)

Task performance is evaluated via an electrical energy model separating:
- motor Joule losses with additional static losses
- gearbox friction losses

The resulting energy metric is returned to the outer loop as the objective for actuator parameter selection.

---
## Status

🚧 **Code release in preparation.**

The implementation will be made publicly available after refactoring and documentation cleanup.

---

## Citation

If you use this work in academic research, please cite:

(BibTeX will be updated after final publication details are available.)
