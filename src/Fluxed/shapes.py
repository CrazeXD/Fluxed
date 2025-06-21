import numpy as np
import scipy.ndimage

from Fluxed.distributions import Distribution

import functools
import warnings

# --- NdShape Class ---


class NdShape:
    """
    A class to represent an N-dimensional shape defined by a border,
    and compute flux through its enclosed region based on an intensity distribution.

    1s in the initial `shape_array` represent border points, 0s represent empty space.
    The 'flux' is the sum of intensity values within the enclosed region, where
    intensity is determined by a provided Distribution object.

    Attributes:
        _shape_array (np.ndarray): The N-dimensional array defining the shape's border.
        _intensity_array (np.ndarray): Stores the computed intensity values across the
                                       entire shape domain, populated by `fill_intensity_array`.
        _current_distribution_id (str): A simple identifier for the last used distribution
                                        for internal caching purposes.

    Properties:
        shape_array (np.ndarray): Returns the underlying NumPy array defining the border.
        dimensions (int): Returns the number of dimensions of the shape.
        is_closed (bool): Checks and caches whether the shape's border encloses a region.
    """

    def __init__(self, shape_array: np.ndarray):
        """
        Initializes the NdShape with a given NumPy array defining the border.

        Args:
            shape_array (np.ndarray): The N-dimensional array representing the shape's border.
                                      1s represent border points, 0s represent empty space.

        Raises:
            TypeError: If the input 'shape_array' is not a NumPy array.
            ValueError: If the array contains values other than 0 or 1.
        """
        if not isinstance(shape_array, np.ndarray):
            raise TypeError("Input 'shape_array' must be a NumPy array.")

        unique_values = np.unique(shape_array)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"Shape array must contain only 0s and 1s. Found: {unique_values}"
            )

        self._shape_array = shape_array
        self._intensity_array = None  # No intensity array initially
        # Tracks the distribution used to fill intensity
        self._current_distribution_id = None

    @property
    def shape_array(self) -> np.ndarray:
        """
        Returns the underlying NumPy array representing the shape's border definition.
        """
        return self._shape_array

    @property
    def dimensions(self) -> int:
        """
        Returns the number of dimensions of the shape.
        """
        return self._shape_array.ndim

    @functools.cached_property
    def is_closed(self) -> bool:
        """
        Checks if the shape's border (1s) completely encloses any region of 0s.
        A shape is considered closed if there is at least one region of 0s
        that is not connected to the array's boundary.

        Returns:
            bool: True if the shape contains at least one enclosed region, False otherwise.
        """
        pass

    def fill_intensity_array(self, distribution: Distribution, *coords_arrays: np.ndarray):
        """
        Fills the shape's domain with intensity values from the given distribution.
        This method computes the `_intensity_array` which will be used for flux calculations.

        Args:
            distribution (Distribution): An instance of the Distribution class.
            *coords_arrays (np.ndarray): Variable number of 1D NumPy arrays, each defining
                the coordinate values along a dimension (x, y, z, ...). The number of arrays
                must match the shape's dimensions. If no coordinate arrays are provided,
                integer indices (0 to dim_size-1) are used for each dimension.

        Raises:
            TypeError: If `distribution` is not a `Distribution` instance.
            ValueError: If the number or shape of `coords_arrays` is incorrect.
        """
        pass

    def get_flux(self, distribution: Distribution, *coords_arrays: np.ndarray) -> float:
        """
        Computes the total flux (sum of intensity values) within the enclosed region
        of the shape, given a distribution of intensity.

        The flux is only well-defined and computed if the shape is closed.

        Args:
            distribution (Distribution): An instance of the Distribution class.
            *coords_arrays (np.ndarray): Variable number of 1D NumPy arrays, each defining
                the coordinate values along a dimension (x, y, z, ...). The number of arrays
                must match the shape's dimensions. If no coordinate arrays are provided,
                integer indices (0 to dim_size-1) are used for each dimension.

        Returns:
            float: The total sum of intensity values within the enclosed region.
                   Returns 0.0 if the shape is not closed, or if SciPy is not available.
                   Returns 0.0 if there is no enclosed region.
        """

        if not self.is_closed:
            warnings.warn(
                "Shape is not closed. Flux is ill-defined for border shapes. "
                "Returning 0.0."
            )
            return 0.0

        # Populate or re-populate the intensity array based on the distribution and coordinates
        self.fill_intensity_array(distribution, *coords_arrays)

        # Retrieve flux from cache using the unique ID generated by fill_intensity_array
        return self._get_flux_internal(self._current_distribution_id)
