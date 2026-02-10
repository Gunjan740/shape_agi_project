
import random
import numpy as np
import math
import torch
import cv2 # Import OpenCV
from tqdm import trange
import warnings
import sys
from pathlib import Path

# Add parent directory to path to import RealTimeEnvironment
sys.path.append(str(Path(__file__).parent.parent))
#from realtime_environment import RealTimeEnvironment
from .realtime_environment import RealTimeEnvironment

warnings.filterwarnings("ignore")

class ShapeEnv(RealTimeEnvironment):
    def __init__(self, noisy=False, noise_value=0.1, device='cpu', time_scaling=1.0, random_position=True, return_states=False,img_size=144,random_initial=False,
                 shapes=['circle', 'triangle', 'square', 'pentagon'],
                 sizes=[15, 30, 45],
                 colors=['red', 'green', 'blue', 'black']):
        # Initialize shape-specific attributes first
        self.shapes_list = shapes
        self.sizes_list = sizes
        self.colors_list = colors
        self.random_initial = random_initial
        if random_initial:
            self.color_idx = random.randint(0, len(self.colors_list)-1)
            self.shape_size_idx = random.randint(0, len(self.shapes_list)*len(self.sizes_list)-1)
        else:
            self.color_idx = 0
            self.shape_size_idx = 0

        self.lim = img_size # Image dimension (width and height)
        self.noisy = noisy
        self.noise_value = noise_value
        self.random_position = random_position

        # Call parent constructor (which will call _get_initial_state)
        super().__init__(device=device, time_scaling=time_scaling, return_states=return_states)

    def _get_color_bgr(self, color_name):
        """Converts color name to BGR tuple for OpenCV."""
        if color_name == 'red':
            return (0, 0, 255)
        elif color_name == 'green':
            return (0, 255, 0)
        elif color_name == 'blue':
            return (255, 0, 0)
        elif color_name == 'black':
            return (0, 0, 0)
        elif color_name == 'white':
            return (255, 255, 255)
        else:
            return (0, 0, 0) # Default to black

    def _draw_shape_opencv(self, shape_name, size, color_name):
        """Draws a shape using OpenCV and returns an RGB image."""
        image = np.ones((self.lim, self.lim, 3), dtype=np.uint8) * 255  # White background (RGB)
        color_bgr = self._get_color_bgr(color_name)
        
        if self.random_position:
            center_x = random.randint(size, self.lim - size)
            center_y = random.randint(size, self.lim - size)
        else:
            center_x = self.lim // 2
            center_y = self.lim // 2

        if shape_name == 'circle':
            cv2.circle(image, (center_x, center_y), size, color_bgr, -1)
        elif shape_name == 'square':
            top_left_x, top_left_y = center_x - size, center_y - size
            bottom_right_x, bottom_right_y = center_x + size, center_y + size
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color_bgr, -1)
        elif shape_name == 'triangle':
            # For a triangle, 'size' can be interpreted as half the base or height for simplicity
            # Let's make it an equilateral triangle
            side_length = int(size * 2 * math.sqrt(3) / 3) # Approximate side length based on circumradius 'size'
            
            p1 = (center_x, center_y - size)
            p2 = (center_x - int(side_length * math.sqrt(3) / 2), center_y + int(side_length / 2))
            p3 = (center_x + int(side_length * math.sqrt(3) / 2), center_y + int(side_length / 2))
            
            triangle_cnt = np.array([p1, p2, p3])
            cv2.drawContours(image, [triangle_cnt], 0, color_bgr, -1)
        elif shape_name == 'pentagon':
            # For a pentagon, 'size' can be interpreted as the radius of the circumcircle
            points = []
            for i in range(5):
                angle = math.radians(18 + i * 72) # Start at 18 degrees to make one side horizontal
                x = center_x + int(size * math.cos(angle))
                y = center_y + int(size * math.sin(angle))
                points.append((x, y))
            pentagon_cnt = np.array(points, dtype=np.int32)
            cv2.fillPoly(image, [pentagon_cnt], color_bgr)
        
        return image

    def generate_observation(self):
        """Generate state as numpy array (internal helper)"""
        shape_idx = (self.shape_size_idx // len(self.sizes_list)) % len(self.shapes_list)
        size_idx = self.shape_size_idx % len(self.sizes_list)
        
        current_color_name = self.colors_list[self.color_idx % len(self.colors_list)]
        current_shape_name = self.shapes_list[shape_idx]
        current_size = self.sizes_list[size_idx]

        state = self._draw_shape_opencv(current_shape_name, current_size, current_color_name)

        state = state.astype(np.float32) / 255.0  # Normalize to [0, 1]
        if self.noisy:
            noise = np.random.normal(0, self.noise_value, state.shape)
            state = state + noise
            state = np.clip(state, 0, 1) # Clip values to stay within [0, 1]

        return state

    def _get_shape_one_hot(self) -> torch.Tensor:
        shape_idx = (self.shape_size_idx // len(self.sizes_list)) % len(self.shapes_list)
        one_hot = torch.zeros(len(self.shapes_list), device=self.device)
        one_hot[shape_idx] = 1.0
        return one_hot

    def _get_size_one_hot(self) -> torch.Tensor:
        size_idx = self.shape_size_idx % len(self.sizes_list)
        one_hot = torch.zeros(len(self.sizes_list), device=self.device)
        one_hot[size_idx] = 1.0
        return one_hot

    def _get_color_one_hot(self) -> torch.Tensor:
        one_hot = torch.zeros(len(self.colors_list), device=self.device)
        one_hot[self.color_idx % len(self.colors_list)] = 1.0
        return one_hot

    def _get_initial_state(self) -> torch.Tensor:
        """Get the initial state of the environment as a PyTorch tensor."""
        if self.random_initial:
            self.color_idx = random.randint(0, len(self.colors_list)-1)
            self.shape_size_idx = random.randint(0, len(self.shapes_list)*len(self.sizes_list)-1)
        else:
            self.color_idx = 0
            self.shape_size_idx = 0
        state_np = self.generate_observation()
        # Convert to torch tensor: shape (H, W, C) -> (C, H, W) for PyTorch convention
        state_tensor = torch.from_numpy(state_np).float()
        state_tensor = state_tensor.permute(2, 0, 1)  # HWC -> CHW
        return state_tensor.to(self.device).unsqueeze(0) # add batch dim

    def _get_state(self) -> torch.Tensor:
        """Get current state of the environment."""
        shape_idx = (self.shape_size_idx // len(self.sizes_list)) % len(self.shapes_list)
        size_idx = self.shape_size_idx % len(self.sizes_list)

        current_color_name = self.colors_list[self.color_idx % len(self.colors_list)]
        current_shape_name = self.shapes_list[shape_idx]
        current_size = self.sizes_list[size_idx]
  
        return {
            "observation" : self.state.clone(),
            "color": current_color_name,
            "shape": current_shape_name,
            "size": current_size,
            "color_one_hot": self._get_color_one_hot(),
            "shape_one_hot": self._get_shape_one_hot(),
            "size_one_hot": self._get_size_one_hot(),
            "step": self.step_count
        }
    
    def _step_simulation(self, action: torch.Tensor) -> torch.Tensor:
        """
        Perform one simulation step.

        Args:
            action: One-hot encoded action vector [5] where:
                - [1,0,0,0,0]: color-
                - [0,1,0,0,0]: color+
                - [0,0,1,0,0]: shape/size-
                - [0,0,0,1,0]: shape/size+
                - [0,0,0,0,1]: no-op (regenerate with new position/noise)

                If action is a scalar, it will be treated as the action index for backwards compatibility.

        Returns:
            observation: New observation after simulation step
        """
        # Handle both one-hot and scalar action formats

        action_int = torch.argmax(action).item() # one hot action

        # Apply action
        if action_int == 0: # color-
            self.color_idx -= 1
        elif action_int == 1: # color+
            self.color_idx += 1
        elif action_int == 2: # shape/size-
            self.shape_size_idx -= 1
        elif action_int == 3: # shape/size+
            self.shape_size_idx += 1
        # else: no-op (action 4 or any other value)
        # Position/noise still changes because generate_state() uses random placement

        # Generate new state (position changes randomly even for no-op)
        state_np = self.generate_observation()
        state_tensor = torch.from_numpy(state_np).float()
        state_tensor = state_tensor.permute(2, 0, 1)  # HWC -> CHW
        self.state = state_tensor.to(self.device)#.unsqueeze(0) # add batch dim
        self.step_count += 1

        return self.state.clone()

    def render(self):
        """
        Render the current state as an image.

        Returns:
            numpy array of the rendered image (HWC, RGB)
        """
        # Convert from CHW to HWC for visualization
        state_np = self.state.cpu().squeeze(0).permute(1, 2, 0).numpy()
        return (state_np * 255).astype(np.uint8)