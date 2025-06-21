import math

# This file contains the implementation of various probability distributions
# Custom distributions can inherit from the Distribution class

class Distribution:
    """
    Callable base class for all probability distributions.
    This class provides a common interface for all distributions.
    It should be inherited by all custom distributions.
    
    Attributes:
        name (str): The name of the distribution.
        func (callable): The function that defines the distribution.
        vars (tuple): The variable names for the parameters of the distribution from func.
    """
    def __init__(self, name: str, func: callable):
        self.name: str = name
        self.func: callable = func
        self.vars = func.__code__.co_varnames[:func.__code__.co_argcount]
        self._validate()
    
    def _validate(self):
        if not callable(self.func):
            raise TypeError(f"{self.name} must be a callable function.")
        if not self.vars:
            raise ValueError(f"{self.name} must have at least one parameter.")
    
    def __call__(self, *args, **kwargs):
        """
        Call the distribution function with the provided arguments.
        
        Args:
            *args: Positional arguments for the distribution function.
            **kwargs: Keyword arguments for the distribution function.
        
        Returns:
            The result of the distribution function.
        """
        return self.func(*args, **kwargs)
    
    def __repr__(self):
        """
        String representation of the distribution.
        
        Returns:
            str: The name of the distribution.
        """
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self):
        """
        String representation of the distribution.
        Returns:
            str: The name of the distribution.
        """
        return self.name
    
class NormalDistribution1D(Distribution):
    """
    Normal distribution in 1D.
    
    Attributes:
        name (str): The name of the distribution.
        func (callable): The function that defines the distribution.
    """
    def __init__(self, mean: float = 0.0, stddev: float = 1.0):
        """
        Initialize the normal distribution with mean and standard deviation.
        
        Args:
            mean (float): The mean of the distribution.
            stddev (float): The standard deviation of the distribution.
        """
        def normal_func(x):
            return (1 / (stddev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / stddev) ** 2)
        
        super().__init__("NormalDistribution1D", normal_func)
        self.mean = mean
        self.stddev = stddev
    
    
