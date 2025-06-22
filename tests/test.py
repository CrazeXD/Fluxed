# main.py
import numpy as np
from Fluxed.shapes import NdShape
from Fluxed.distributions import NormalDistribution2D, LinearDistribution1D, UniformDistribution
from Fluxed.match import match_flux_parameters

print("--- 3D HARDER TEST CASE ---")

# --- 1. Define the SOURCE shape and distribution ---
source_shape_2d_border = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=int)
source_shape = NdShape(source_shape_2d_border)
source_dist = NormalDistribution2D(
    mean_x=1.5, mean_y=4.5, stddev_x=2.0, stddev_y=2.0)
source_x_coords = np.linspace(-10, 10, 7)
source_y_coords = np.linspace(-10, 10, 7)
source_coords = (source_x_coords, source_y_coords)

print(
    f"Source Shape: 2D Donut, Enclosed Area = {source_shape.get_flux(UniformDistribution(1.0)):.0f} points")

# --- 2. Define the TARGET shape and distribution ---
target_shape_3d_border = np.ones((5, 5, 5), dtype=int)
target_shape_3d_border[1:4, 1:4, 1:4] = 0
target_shape = NdShape(target_shape_3d_border)
target_x_coords = np.arange(5)
target_y_coords = np.arange(5)
target_z_coords = np.linspace(100, 200, 5)
target_coords = (target_x_coords, target_y_coords, target_z_coords)

print(
    f"Target Shape: 3D Hollow Cube, Enclosed Volume = {target_shape.get_flux(UniformDistribution(1.0)):.0f} points")

# --- CORRECTED WRAPPER CLASS ---


class LinearDistribution1D_on_Z(LinearDistribution1D):
    """
    A wrapper that adapts a 1D Linear Distribution to a 3D space
    by making it dependent only on the 'z' coordinate.
    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        # Initialize the parent 1D distribution as normal.
        # This sets up the original self.func = linear_func(x).
        super().__init__(slope=slope, intercept=intercept)

    def __call__(self, x, y, z):
        """
        This is the new callable for the 3D space.
        It ignores x and y, and passes z to the parent's 1D function.
        This method will be called directly by NdShape.
        """
        # self.func is the original 1D function from the parent.
        # It's already vectorized to handle a NumPy array 'z'.
        return self.func(z)


TargetDistClass = LinearDistribution1D_on_Z
params_to_find = ['slope', 'intercept']

# --- 3. Set up the Optimization ---
initial_parameter_guess = [0.0, 1.0]
param_bounds = [
    (0.001, None),
    (-10.0, 10.0)
]
optimizer_options = {'disp': True, 'ftol': 1e-9}

# --- 4. Run the matching function! ---
optimization_result = match_flux_parameters(
    source_shape=source_shape,
    source_dist=source_dist,
    target_shape=target_shape,
    TargetDistClass=TargetDistClass,
    param_names=params_to_find,
    initial_guess=initial_parameter_guess,
    source_coords_arrays=source_coords,
    target_coords_arrays=target_coords,
    bounds=param_bounds,
    optimizer_options=optimizer_options
)

# --- 5. Print and verify the results ---
print("\n--- OPTIMIZATION FINISHED ---")
print(f"Success: {optimization_result['success']}")
print(f"Optimizer Message: {optimization_result['message']}")
print(f"\nTarget Flux Was: {optimization_result['target_flux']:.4f}")
print(f"Final Flux Achieved: {optimization_result['final_flux']:.4f}")
print(f"Found Optimal Parameters: {optimization_result['parameters']}")

if optimization_result['success']:
    found_params = optimization_result['parameters']
    final_dist = TargetDistClass(**found_params)
    verification_flux = target_shape.get_flux(final_dist, *target_coords)

    print(
        f"\nVerification flux with found parameters: {verification_flux:.4f}")

    if np.isclose(verification_flux, optimization_result['target_flux']):
        print("Verification PASSED!")
    else:
        print("Verification FAILED (fluxes do not match closely enough).")
        print(
            f"Difference: {abs(verification_flux - optimization_result['target_flux']):.6f}")
