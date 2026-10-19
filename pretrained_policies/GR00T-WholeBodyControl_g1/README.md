# GR00T Locomotion Policy Architecture

**Model**: GR00T-WholeBodyControl-Walk  
**Source**: NVIDIA GR00T Whole Body Control  
**Total Parameters**: 470,578  
---

## Table of Contents
1. [Overview](#overview)
2. [Input Space (Observation)](#input-space-observation)
3. [Output Space (Action)](#output-space-action)
4. [Network Architecture](#network-architecture)
5. [Forward Pass Data Flow](#forward-pass-data-flow)
6. [Comparison with Unitree Policy](#comparison-with-unitree-policy)
7. [Training Considerations](#training-considerations)

---

## Overview

The GR00T policy is an **asymmetric actor-critic** architecture designed for bipedal locomotion with whole-body control. It uses:
- A **temporal observation stack** (6 frames × 86D = 516D)
- An **estimator network** (encoder) that processes full observations into a compressed latent space
- An **actor network** (decoder) that generates actions from the latent representation + recent observations

### Key Features
- **Temporal context**: 6-frame history for stability and dynamics understanding
- **Privileged learning**: Estimator sees all 29 DOF, actor only controls 15 DOF (legs + waist)
- **Height & orientation commands**: Explicit control over base height and orientation
- **ELU activations**: For smoother gradients compared to ReLU

---

## Input Space (Observation)

### Single Frame (86D)
One observation frame contains:

```
[0:3]     velocity_commands (3D)     - [vx, vy, yaw_rate] desired velocities
[3]       height_command (1D)        - desired base height
[4:7]     orientation_command (3D)   - [roll, pitch, yaw] desired orientation
[7:10]    angular_velocity (3D)      - IMU angular velocity (scaled by 0.5)
[10:13]   gravity_vector (3D)        - projected gravity direction
[13:42]   joint_positions (29D)      - all joint angles relative to default pose
[42:71]   joint_velocities (29D)     - all joint velocities
[71:86]   previous_actions (15D)     - actions from previous timestep
---
Total: 86D per frame
```

### Stacked Observation (516D)
The policy receives **6 consecutive frames** stacked together:

```
obs_516d = [frame_t-5, frame_t-4, frame_t-3, frame_t-2, frame_t-1, frame_t]
         = 6 × 86D = 516D
```

**Temporal reasoning**: The history allows the policy to:
- Infer velocities and accelerations
- Predict contact events and transitions
- Maintain gait stability across steps

### Observation Scaling
```python
# From Unitree training config
ang_vel_scale = 0.5          # Angular velocity scaling
dof_pos_scale = 1.0          # Joint position scaling  
dof_vel_scale = 0.05         # Joint velocity scaling
cmd_scale = [2.0, 2.0, 0.5]  # [vx, vy, yaw] command scaling
```

### Default Joint Positions (Standing Pose)
```python
default_angles = [
    # Left leg (6D)
    -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    
    # Right leg (6D)  
    -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,
    
    # Waist (3D)
    0.0, 0.0, 0.0,  # yaw, roll, pitch
    
    # Left arm (7D)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
    
    # Right arm (7D)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]
```

---

## Output Space (Action)

### Action Vector (15D)
```
[0:6]     left_leg_deltas (6D)      - position offsets for left leg joints
[6:12]    right_leg_deltas (6D)     - position offsets for right leg joints
[12:15]   waist_deltas (3D)         - position offsets for waist joints
---
Total: 15D
```

### Action Scaling & Application
```python
action_scale = 0.25  # Action multiplier

# Target joint positions
target_pos = default_angles[:15] + action * action_scale
```

### Controlled DOFs
The policy controls **only the lower body**:
- **12 leg joints**: Full leg control for locomotion
- **3 waist joints**: Yaw, roll, pitch for balance

The **arms (14 DOF)** are NOT controlled by the locomotion policy and can be:
- Held at fixed positions (e.g., slight forward/outward pose)
- Controlled by a separate arm policy
- Used for manipulation while walking

---

## Network Architecture

### Overview Diagram
```
Input (516D)
    │
    ├─────────────────────────────────────────────────┐
    │                                                  │
    v                                                  v
┌─────────────────┐                          ┌────────────────┐
│   ESTIMATOR     │                          │  Input Slice   │
│   (Encoder)     │                          │   (Last 86D)   │
└─────────────────┘                          └────────────────┘
    │                                                  │
    │ (35D latent)                                     │
    │                                                  │
    ├──► [:3]     ──────────┐                        │
    │    (3D)               │                        │
    │                       v                        v
    └──► [3:]     ────► Normalize  ─────►  Concat (121D)
         (32D)            (32D)                      │
                                                     v
                                             ┌─────────────┐
                                             │    ACTOR    │
                                             │  (Decoder)  │
                                             └─────────────┘
                                                     │
                                                     v
                                               Output (15D)
```

---

## 1. Estimator Network (Encoder)

**Purpose**: Encode full 516D observation into a compact 35D latent representation.

### Architecture
```python
Estimator(
    # Layer 1: Initial compression
    Linear(516, 256),      # Weight: [256, 516] = 132,096 params
    ELU(),                 # Smooth activation
    
    # Layer 2: Feature refinement  
    Linear(256, 256),      # Weight: [256, 256] = 65,536 params
    ELU(),
    
    # Layer 3: Final encoding
    Linear(256, 35),       # Weight: [35, 256] = 8,960 params
)
```

### Layer Details

#### Layer 1: `Linear(516 → 256)` + `ELU()`
- **Input**: 516D stacked observations (6 frames × 86D)
- **Output**: 256D intermediate features
- **Parameters**: 132,096 + 256 (bias) = 132,352
- **Purpose**: Initial dimensionality reduction while preserving temporal information

#### Layer 2: `Linear(256 → 256)` + `ELU()`
- **Input**: 256D intermediate features
- **Output**: 256D refined features  
- **Parameters**: 65,536 + 256 (bias) = 65,792
- **Purpose**: Feature refinement and non-linear transformation

#### Layer 3: `Linear(256 → 35)`
- **Input**: 256D refined features
- **Output**: 35D latent encoding
- **Parameters**: 8,960 + 35 (bias) = 8,995
- **Purpose**: Final compression to latent space

**Total Estimator Parameters**: 207,139

### Latent Space Breakdown (35D)
The 35D latent is split into two components:

1. **First 3D** (indices 0-2): Direct features (not normalized)
2. **Last 32D** (indices 3-34): Dynamics features (L2-normalized for stability)

**Normalization**:
```python
latent_first = latent[:, :3]     # (batch, 3)
latent_rest = latent[:, 3:]      # (batch, 32)

# L2 normalize the rest
latent_rest_norm = torch.clamp(
    torch.norm(latent_rest, p=2, dim=-1, keepdim=True),
    min=1e-8
)
latent_rest_normalized = latent_rest / latent_rest_norm  # (batch, 32)
```

---

## 2. Actor Network (Decoder)

**Purpose**: Decode latent representation + recent observation into motor actions.

### Actor Input Construction (121D)
```python
actor_input = torch.cat([
    obs_recent,             # Last 86D from input (most recent frame)
    latent_first,           # First 3D from estimator (direct features)
    latent_rest_normalized  # Last 32D from estimator (normalized dynamics)
], dim=-1)

# Total: 86D + 3D + 32D = 121D
```

### Architecture
```python
Actor(
    # Layer 1: Initial expansion
    Linear(121, 512),      # Weight: [512, 121] = 61,952 params
    ELU(),
    
    # Layer 2: Feature processing
    Linear(512, 256),      # Weight: [256, 512] = 131,072 params
    ELU(),
    
    # Layer 3: Feature refinement
    Linear(256, 256),      # Weight: [256, 256] = 65,536 params
    ELU(),
    
    # Layer 4: Action head
    Linear(256, 15),       # Weight: [15, 256] = 3,840 params
)
```

### Layer Details

#### Layer 1: `Linear(121 → 512)` + `ELU()`
- **Input**: 121D (recent obs + latent)
- **Output**: 512D expanded features
- **Parameters**: 61,952 + 512 (bias) = 62,464
- **Purpose**: Expand compressed representation for action generation

#### Layer 2: `Linear(512 → 256)` + `ELU()`
- **Input**: 512D expanded features
- **Output**: 256D intermediate features
- **Parameters**: 131,072 + 256 (bias) = 131,328
- **Purpose**: Process features with large capacity

#### Layer 3: `Linear(256 → 256)` + `ELU()`
- **Input**: 256D intermediate features
- **Output**: 256D refined features
- **Parameters**: 65,536 + 256 (bias) = 65,792
- **Purpose**: Additional refinement layer for complex motor patterns

#### Layer 4: `Linear(256 → 15)`
- **Input**: 256D refined features
- **Output**: 15D actions (leg + waist deltas)
- **Parameters**: 3,840 + 15 (bias) = 3,855
- **Purpose**: Final action prediction layer

**Total Actor Parameters**: 263,439

---

## Forward Pass Data Flow

### Complete Forward Pass
```python
def forward(obs_516d):
    """
    Args:
        obs_516d: (batch, 516) stacked observations
        
    Returns:
        actions: (batch, 15) joint position deltas
    """
    # 1. Encode full observation
    latent = estimator(obs_516d)  # (batch, 35)
    
    # 2. Split and normalize latent
    latent_first = latent[:, :3]              # (batch, 3)
    latent_rest = latent[:, 3:]               # (batch, 32)
    
    latent_rest_norm = torch.clamp(
        torch.norm(latent_rest, p=2, dim=-1, keepdim=True),
        min=1e-8
    )
    latent_rest_normalized = latent_rest / latent_rest_norm  # (batch, 32)
    
    # 3. Extract recent observation
    obs_recent = obs_516d[:, -86:]  # Last frame (batch, 86)
    
    # 4. Concatenate actor input
    actor_input = torch.cat([
        obs_recent,              # 86D
        latent_first,            # 3D
        latent_rest_normalized   # 32D
    ], dim=-1)  # Total: 121D
    
    # 5. Generate actions
    actions = actor(actor_input)  # (batch, 15)
    
    return actions
```

### Computational Cost
```
Forward pass operations:
1. Estimator: 516 → 256 → 256 → 35     (~207K params)
2. Normalization: L2 norm on 32D       (~32 ops)
3. Concatenation: Build 121D input     (memory copy)
4. Actor: 121 → 512 → 256 → 256 → 15  (~263K params)

Total: ~470K multiply-adds per inference
```

## References

1. **GR00T Project**: https://github.com/NVlabs/gr00t_wbc
2. **Paper**: "GR00T: Whole-Body Control for Humanoid Robots" (NVIDIA Research)
3. **Unitree SDK**: https://github.com/unitreerobotics/unitree_sdk2_python

---

## License

The GR00T model is provided by NVIDIA under the NVIDIA Open Model License.  
See `NVIDIA Open Model License` in the model directory for details.

---

**Last Updated**: 2025-01-20  

