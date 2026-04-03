from actorob.models import ModelFactory, expand_config
from actorob.actuators import ActuatorUnit, ActuatorPosition
from actorob.utils import render

# ---------------------------------------------------------------------------
# Define actuator configurations for different gear ratios and torque classes
# ---------------------------------------------------------------------------

G12_T150 = ActuatorUnit(
    name="G12-T150",
    gear_ratio=12.0,
    nominal_torque=54.0,
    max_torque=150.0,
    nominal_velocity=121.0,
    max_velocity=140.0,
    rotor_inertia=0.000427,
    damping=0.0069,
    friction_loss=0.576,
    torque_constant=2.5,
    resistance=0.236,
    voltage=48,
    diameter=0.12,
    length=0.0641,
    mass=1.35,
)

G24_360 = ActuatorUnit(
    name="G24-360",
    gear_ratio=24.0,
    nominal_torque=100.0,
    max_torque=360.0,
    nominal_velocity=102.0,
    max_velocity=110.0,
    rotor_inertia=0.00056,
    torque_constant=2.6,
    resistance=0.0796,
    voltage=48,
    diameter=0.124,
    length=0.085,
    mass=2.18,
)

G11_T380 = ActuatorUnit(
    name="G11-T380",
    gear_ratio=11.4,
    nominal_torque=120.0,
    max_torque=380.0,
    nominal_velocity=105.0,
    max_velocity=130.0,
    rotor_inertia=0.0012,
    torque_constant=2.5,
    resistance=0.0655,
    voltage=48,
    diameter=0.17,
    length=0.083,
    mass=3.2,
)

G18_T90 = ActuatorUnit(
    name="G18-T90",
    gear_ratio=18.0,
    nominal_torque=30.0,
    max_torque=90.0,
    nominal_velocity=142.0,
    max_velocity=145.0,
    rotor_inertia=0.000134,
    damping=0.028,
    friction_loss=0.876,
    torque_constant=2.1,
    resistance=0.315,
    voltage=48,
    diameter=0.105,
    length=0.054,
    mass=0.84,
)

G36_T36 = ActuatorUnit(
    name="G36-T36",
    gear_ratio=36.0,
    nominal_torque=12.0,
    max_torque=36.0,
    nominal_velocity=78.0,
    max_velocity=93.0,
    rotor_inertia=1.38e-05,
    damping=0.014,
    friction_loss=0.23,
    torque_constant=1.4,
    resistance=0.567,
    voltage=48,
    diameter=0.056,
    length=0.0605,
    mass=0.377,
)


# ----------------------------------------------------
# Define actuator mappings for robot body subsystems
# ----------------------------------------------------

LEGS = {
    "hip_pitch": G11_T380,
    "hip_roll": G24_360,
    "hip_yaw": G12_T150,
    "knee_pitch": G11_T380,
    "ankle_pitch": G18_T90,
    "ankle_roll": G18_T90,
}

TORSO = {
    "torso_yaw": G12_T150,
}

ARMS = {
    "shoulder_pitch": G18_T90,
    "shoulder_roll": G18_T90,
    "shoulder_yaw": G36_T36,
    "elbow_pitch": G36_T36,
    "elbow_yaw": G36_T36,
}


# ----------------------------------------------------
# Build the complete actuator configuration dictionary
# ----------------------------------------------------

ACTUATORS_SET = expand_config(LEGS)
ACTUATORS_SET = expand_config(TORSO, ACTUATORS_SET, mirror=False)
ACTUATORS_SET = expand_config(ARMS, ACTUATORS_SET)


# ----------------------------------------------------
# Example 1: Load a robot model where actuators are already defined as extra bodies,
# and apply actuator parameters to update them.
# ----------------------------------------------------

# Load the robot XML model. If the robot already defines actuators as additional bodies,
# ModelFactory will automatically update their properties using the provided parameters.
xml_path = "robots/humanoid_pry/humanoid_pry.xml"
robot = ModelFactory(xml_path, ACTUATORS_SET)

# Add a predefined scene (e.g., ground plane, lighting, environment objects).
robot.add_scene()

# Compile the MuJoCo model and create the associated runtime data structures.
model, data = robot.build()

# Render the model to visually inspect actuator placement and configuration.
render(model, data)

# Save the updated XML model with actuator parameters applied.
robot.save()

# ----------------------------------------------------
# Example 2: Load a robot model without actuator bodies and create them programmatically
# using actuator parameters and explicit placement information.
# ----------------------------------------------------

# Define actuator placement (position/orientation) and, optionally, the parent body.
# If a parent body is not provided, the default parent is inferred from the joint hierarchy
# (typically the parent of the body that contains the joint).
ACTUATORS_POSITIONS = {
    "left_hip_pitch_joint": ActuatorPosition(
        pos=[0.00024134, 0.07135611, -0.11391088], quat=[0.9659, -0.2588, 0.0, 0.0]
    ),
    "right_hip_pitch_joint": ActuatorPosition(
        pos=[-0.00044405, -0.0715722, -0.11353501], quat=[0.9659, 0.2588, 0.0, 0.0]
    ),
    "left_ankle_pitch_joint": ActuatorPosition(
        pos=[-0.00015591, 0.0004461, -0.10010125], parent_body_name="left_knee_pitch_link"
    ),
    "left_ankle_roll_joint": ActuatorPosition(
        pos=[-0.00031957, -0.0004459, -0.20996094], quat=[0, 0.7071, 0.7071, 0], parent_body_name="left_knee_pitch_link"
    ),
    "left_hip_yaw_joint": ActuatorPosition(pos=[5.2e-07, 3e-08, 0.0224909], parent_body_name="left_hip_yaw_link"),
    "left_knee_pitch_joint": ActuatorPosition(
        pos=[-0.00023836, -0.0084458, -0.16797885], parent_body_name="left_hip_yaw_link"
    ),
    "left_hip_roll_joint": ActuatorPosition(
        pos=[-0.0057416, 0.07997545, -0.03101416],
        quat=[0.9659, 0.2588, 0.0, 0.0],
        parent_body_name="left_hip_pitch_link",
    ),
    "right_ankle_pitch_joint": ActuatorPosition(
        pos=[-0.0015008, -0.0003883, -0.09989582], parent_body_name="right_knee_pitch_link"
    ),
    "right_ankle_roll_joint": ActuatorPosition(
        pos=[-0.00156307, 0.0005039, -0.21004613], quat=[0, 0.7071, 0.7071, 0], parent_body_name="right_knee_pitch_link"
    ),
    "right_hip_yaw_joint": ActuatorPosition(
        pos=[-0.00146102, 5.766e-05, 0.0224909], parent_body_name="right_hip_yaw_link"
    ),
    "right_knee_pitch_joint": ActuatorPosition(
        pos=[-0.00159905, 0.0085035, -0.16780417], parent_body_name="right_hip_yaw_link"
    ),
    "right_hip_roll_joint": ActuatorPosition(
        pos=[-0.0057416, -0.08002355, -0.0309864],
        quat=[0.9659, -0.2588, 0.0, 0.0],
        parent_body_name="right_hip_pitch_link",
    ),
    "left_elbow_yaw_joint": ActuatorPosition(
        pos=[-2.69e-06, 0.02900721, -0.099498], parent_body_name="left_elbow_pitch_link"
    ),
    "left_shoulder_yaw_joint": ActuatorPosition(
        pos=[5.24e-06, -5.63e-06, 0.026002], parent_body_name="left_shoulder_yaw_link"
    ),
    "left_elbow_pitch_joint": ActuatorPosition(
        pos=[-2.69e-06, -0.003998, -0.12800721], parent_body_name="left_shoulder_yaw_link"
    ),
    "left_shoulder_roll_joint": ActuatorPosition(
        pos=[-0.003946, 0.06989856, -4.593e-05], parent_body_name="left_shoulder_pitch_link"
    ),
    "left_shoulder_pitch_joint": ActuatorPosition(
        pos=[0.00011129, 0.147954, 0.35499639], parent_body_name="torso_yaw_link"
    ),
    "right_elbow_yaw_joint": ActuatorPosition(
        pos=[-2.69e-06, -0.02899279, -0.099498], parent_body_name="right_elbow_pitch_link"
    ),
    "right_shoulder_yaw_joint": ActuatorPosition(
        pos=[-5.24e-06, 5.63e-06, 0.026002], parent_body_name="right_shoulder_yaw_link"
    ),
    "right_elbow_pitch_joint": ActuatorPosition(
        pos=[2.69e-06, 0.003998, -0.12800721], parent_body_name="right_shoulder_yaw_link"
    ),
    "right_shoulder_roll_joint": ActuatorPosition(
        pos=[-0.003946, -0.07010144, -4.593e-05], parent_body_name="right_shoulder_pitch_link"
    ),
    "right_shoulder_pitch_joint": ActuatorPosition(
        pos=[-0.00011129, -0.147954, 0.35499639], parent_body_name="torso_yaw_link"
    ),
    "torso_yaw_joint": ActuatorPosition(pos=[5.2e-07, -5e-08, 0.0250091]),
}

# Load the robot XML model. Since actuators are not pre-defined as extra bodies,
# ModelFactory will create actuator bodies using ACTUATORS_SET and ACTUATORS_POSITIONS.
xml_path = "robots/humanoid_pry/humanoid_pry_merged.xml"
robot = ModelFactory(xml_path, ACTUATORS_SET, ACTUATORS_POSITIONS)

# Add a predefined scene (e.g., ground plane, lighting, environment objects).
robot.add_scene()

# Compile the MuJoCo model and create the associated runtime data structures.
model, data = robot.build()

# Render the model to visually inspect actuator placement and configuration.
render(model, data)

# ----------------------------------------------------
# Example 3: Replace several actuators with a new type
# ----------------------------------------------------

ACTUATORS_SET["left_hip_roll_joint"] = G11_T380
ACTUATORS_SET["left_ankle_roll_joint"] = G11_T380
ACTUATORS_SET["left_shoulder_pitch_joint"] = G11_T380

# Apply updated actuator configuration
robot.set_actuators(ACTUATORS_SET)

# Rebuild and render again to visualize changes
model, data = robot.build()
render(model, data)

# ----------------------------------------------------
# Example 4: Assign consistent colors to similar actuators
# ----------------------------------------------------

# Group similar actuators and assign the same color to each group
robot.colorize_similar_actuators()

# Rebuild the model and render it again to visualize the updated actuator colors
model, data = robot.build()
render(model, data)

# Save the updated version of the model with the new actuator color assignments
robot.save()
