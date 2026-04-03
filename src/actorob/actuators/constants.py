# -- Motor regression coefficients (mass m now in kilograms) --
# Motor rotor inertia regression: J = A * m^B [kg*m^2], m in kg
ROTOR_INERTIA_COEFF_A = 3.455989677971238e-4
ROTOR_INERTIA_COEFF_B = 1.2679580131364627

# Motor peak torque regression: T = A1 * m + A2 * m^2 + B [Nm], m in kg
PEAK_MOTOR_TORQUE_COEFF_A1 = 11.2203505
PEAK_MOTOR_TORQUE_COEFF_A2 = 1.24542702
PEAK_MOTOR_TORQUE_COEFF_B = 0.24050658010407666

# Motor no-load velocity regression: w = A * m^B [rad/s], m in kg
NO_LOAD_VELOCITY_COEFF_A = 351.6641004113412
NO_LOAD_VELOCITY_COEFF_B = -0.2

# Motor constant regression: Km = A * m^B [Nm/A], m in kg
MOTOR_CONSTANT_COEFF_A = 0.6782622892336895
MOTOR_CONSTANT_COEFF_B = 0.7098168640924966

# Motor stator diameter regression: d = A * m^B [m], m in kg
MOTOR_S_DIAMETER_COEFF_A = 0.10204446113377217
MOTOR_S_DIAMETER_COEFF_B = 0.2486277588027095

# Motor nominal/peak torque ratio
PEAK_TO_NOMINAL_TORQUE_RATIO = 0.3

# Motor loss ratio (additional losses / copper losses at nominal conditions)
MOTOR_EXTRA_LOSS_RATIO = 0.42

# Motor no-load/nominal velocity ratio
NO_LOAD_TO_NOMINAL_VELOCITY_RATIO = 0.8

# -- Motor physical parameters --
RHO_STEEL = 7850  # Density of steel [kg/m^3]
RHO_EFF = 3600  # Density of effective rotor material [kg/m^3]
ALPHA = 0.5  # Ratio between outer and inner radius of rotor

# -- Actuator parameters margins --
MARGIN_LENGTH = 1.1  # Length margin to add to actuator length [m]
MARGIN_DIAMETER = 1.2  # Diameter margin to add to actuator diameter [m]
MARGIN_MASS = 1.2  # Mass margin to add to actuator mass [m]

# -- Gearbox parameters --
ONE_STAGE_RATIO = 7  # Gear ratio for one stage gearbox
EFFICIENCY_SLOPE = -0.03  # Slope of efficiency vs gear ratio linear model
EFFICIENCY_INTERCEPT = 0.98  # Intercept of efficiency vs gear ratio linear model
GEARBOX_VOLUME_COEFF_A = 0.02921894  # Coefficient A for gearbox volume regression
GEARBOX_VOLUME_COEFF_B = 2.98837534  # Coefficient B for gearbox volume regression
GEARBOX_LENGTH_COEFF_A = 0.2865  # Coefficient A for gearbox length regression
GEARBOX_LENGTH_COEFF_B = 0.9706  # Coefficient B for gearbox length regression
