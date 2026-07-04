# misarta-py

Python bindings for [misarta](../misarta) — a Rust rigid-body dynamics library
(Pinocchio-equivalent).

## Build (development)

```bash
cd articara/misarta-py
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --release
pytest tests/
```

## Quick start

```python
import numpy as np
import misarta

model = misarta.load_urdf("path/to/robot.urdf")
print(model.nq, model.nv, model.joint_names())

q = np.zeros(model.nq)
data = misarta.forward_kinematics(model, q)
T_world = data.oMi(1)         # SE3 of joint 1 in world frame

J = misarta.compute_joint_jacobian(model, q, joint_id=1)
M = misarta.crba(model, q)                # mass matrix
tau = misarta.rnea(model, q, np.zeros(model.nv), np.zeros(model.nv))
ddq = misarta.aba(model, q, np.zeros(model.nv), tau)
```

See [`../misarta/doc/python-binding-plan.md`](../misarta/doc/python-binding-plan.md)
for the full design.
