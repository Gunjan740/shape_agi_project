import time
import torch

from typing import Callable, Dict, List
from abc import ABC, abstractmethod

class RealTimeEnvironment(ABC):
    """
    Base class for real-time RL environments.
    
    The environment progresses based on actual computation time:
    - Benchmark policy defines "real-time" (e.g., 100ms = 1 simulation step)
    - Slower policies cause more environment steps
    - Faster policies cause fewer environment steps
    """
    
    def __init__(self, device: str = 'cpu', time_scaling: float = 1.0, return_states=False):
        """
        Args:
            device: PyTorch device ('cpu' or 'cuda')
            time_scaling: Scaling factor for time progression (default: 1.0)
        """
        self.device = device
        self.time_scaling = time_scaling
        
        # Benchmark results
        self.benchmark_policy_time = -1  # Time for benchmark policy (ms)
        self.simulation_step_time = -1   # Time for one simulation step (ms)
        
        # Environment state
        self.return_states = return_states
        self.state = self._get_initial_state()
        self.step_count = 0

    @abstractmethod
    def _get_state(self) -> torch.Tensor:
        """Get current state of the environment. This is the ACTUAL state, containing all relevant information."""
        pass
    @abstractmethod
    def _get_initial_state(self) -> torch.Tensor:
        """Get the initial state of the environment."""
        pass
    
    @abstractmethod
    def _step_simulation(self, action: torch.Tensor) -> torch.Tensor:
        """
        Perform one simulation step.
        
        Args:
            action: The action to apply
            
        Returns:
            observation: New observation after simulation step
        """
        pass
    def benchmark_policy(self, policy_fn: Callable[[torch.Tensor], torch.Tensor], num_trials: int = 10):
        """
        Set and benchmark the policy function to define "real-time" baseline.   
        """

        self.benchmark_policy_fn = policy_fn
        self.num_benchmark_trials = num_trials
        return self._benchmark_policy(self.benchmark_policy_fn, num_trials=self.num_benchmark_trials)

    
    def _benchmark_policy(self, policy_fn: Callable[[torch.Tensor], torch.Tensor], num_trials: int = 10) -> float:
        """
        Benchmark a policy to define "real-time" baseline.
        
        Args:
            policy_fn: Policy function to benchmark
            num_trials: Number of trials to average over
            
        Returns:
            Average policy computation time in milliseconds
        """
        times = []
        dummy_obs = self.state.clone()
        
        for _ in range(num_trials):
            start_time = time.perf_counter()
            policy_fn(dummy_obs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        self.benchmark_policy_time = sum(times) / len(times)
        return self.benchmark_policy_time
    
    def benchmark_simulation(self, num_trials: int = 100) -> float:
        """
        Benchmark simulation step time.
        
        Args:
            num_trials: Number of simulation steps to time
            
        Returns:
            Average simulation step time in milliseconds
        """
        # Save current state
        saved_state = self.state.clone()
        saved_step_count = self.step_count
        
        dummy_action = torch.zeros(1, device=self.device)
        
        start_time = time.perf_counter()
        for _ in range(num_trials):
            self._step_simulation(dummy_action)
        end_time = time.perf_counter()
        
        # Restore state
        self.state = saved_state
        self.step_count = saved_step_count
        
        total_time = (end_time - start_time) * 1000  # Convert to ms
        self.simulation_step_time = total_time / num_trials
        return self.simulation_step_time
    
    def _compute_environment_steps(self, actual_policy_time: float) -> int:
        """
        Compute how many environment steps should occur based on policy time.
        
        Args:
            actual_policy_time: Actual time taken by policy (ms)
            
        Returns:
            Number of environment steps to execute
        """
        # Real-time mode: compute steps based on timing
        if self.benchmark_policy_time == -1 and self.time_scaling > 0:
            raise ValueError("Must run benchmark_policy() first for real-time mode")
        
        # How many "real-time units" did this policy take?
        time_ratio = actual_policy_time / self.benchmark_policy_time
        
        # Apply scaling factor
        scaled_steps = time_ratio * self.time_scaling
        
        return max(1, int(scaled_steps))
    
    def warmup(self, policy_fn: Callable, num_warmup_steps: int = 3):
        """Run warmup steps to initialize CUDA and compile kernels."""
        for _ in range(num_warmup_steps):
            with torch.no_grad():
                _ = policy_fn(self.state.clone())
    
    # Synchronize to ensure all operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def step(self, policy_fn: Callable[[torch.Tensor], torch.Tensor], benchmark=False) -> tuple[List[torch.Tensor], Dict]:
        """
        Execute environment step with real-time progression.
        
        Args:
            policy_fn: Policy function that takes observation and returns action
            
        Returns:
            observations: List of all observations during policy computation
            info: Dictionary with step information
        """
        # Time the policy computation
        if self.benchmark_policy_time != -1 and benchmark:
            self._benchmark_policy(self.benchmark_policy_fn, num_trials=self.num_benchmark_trials)
        start_time = time.perf_counter()
        action = policy_fn(self.state.clone())
        end_time = time.perf_counter()
        
        actual_policy_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Compute number of environment steps based on policy time
        num_env_steps = self._compute_environment_steps(actual_policy_time)
        
        # Execute environment steps and collect observations
        observations = []
        states = []
        sim_times = []
        for i in range(num_env_steps): # action repeat

            start_time = time.perf_counter()
            obs = self._step_simulation(action)
            end_time = time.perf_counter()
            actual_sim_time = (end_time - start_time) * 1000 
            sim_times.append(actual_sim_time)
            observations.append(obs)
            if self.return_states:
                states.append(self._get_state())
        

        observations = torch.stack(observations, dim=0) # (num_env_steps, obs_dim)
        # Return info
        info = {
            "num_environment_steps": num_env_steps,
            "actual_policy_time_ms": actual_policy_time,
            "benchmark_policy_time_ms": self.benchmark_policy_time,
            "simulation_step_time_ms": self.simulation_step_time,
            "actual_simulation_time_ms": sum(sim_times) / num_env_steps,
            "time_scaling": self.time_scaling,
            "step_count": self.step_count,
            "performed_action" : action
        }
        if self.return_states:
            return observations, states, info
        
        self.state = observations # list of observations now
        return observations, info

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state"""
        self.state = self._get_initial_state()
        self.step_count = 0
        return self.state.clone()


