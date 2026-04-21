import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises, lognorm
from scipy.stats import vonmises as scipy_vonmises
from scipy.stats import lognorm as scipy_lognorm
import pickle
import jax.numpy as jnp
import pickle as pkl
import sys

sys.path.append('./')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def load_workspace(workspace_path):
    if os.path.exists(workspace_path):
        print(f"Loading OT workspace from {workspace_path}")
        try:
            with open(workspace_path, "rb") as f:
                ws = pkl.load(f)
            print("Workspace loaded successfully")
            return ws
        except Exception as e:
            print(f"Error loading workspace: {e}")
    return None

def generate_pushforward(workspace, all_init_xs, num_points=20):
    condition_vectors = all_init_xs[:,2:]
    unique_conditions = jnp.unique(condition_vectors, axis=0)
    num_conditions = unique_conditions.shape[0]
    paths = []
    for c_idx in range(num_conditions):
        cond_paths=[]
        current_condition = unique_conditions[c_idx]
        condition_matches = jnp.all(condition_vectors == current_condition, axis=1)
        condition_indices = jnp.where(condition_matches)[0]
        init_xs_condition = all_init_xs[condition_indices]

        for i in range(init_xs_condition.shape[0]):
            init_x = init_xs_condition[i]
            x = init_x
            for t in range(2):
                prev_x = x
                x = workspace.neural_dual_solver.pushforward_jit(
                    workspace.state_source_maps[t].params,
                    workspace.state_target_potentials[t].params,
                    workspace.params_geometry,
                    x
                ).solution
                path = workspace.neural_dual_solver.path_jit(
                    workspace.params_geometry, prev_x, x, num_points=num_points)
                cond_paths.append(path)
        paths.append(cond_paths)
    return paths

def generate_conditional_semicircles(num_points_per_condition: int,
                                     num_time_points: int = 6,
                                     radius: float = 1.0,
                                     angular_concentration: float = 15.0,
                                     radial_std_dev: float = 0.1,
                                     device: torch.device = DEVICE):
    """
    Generates 4 conditions of warped semicircle data with independent sampling for each condition.
    Each condition moves along a different semicircle:
    - Condition 0: Top half of circle centered at (-1,0), moving from origin to left
    - Condition 1: Bottom half of circle centered at (-1,0), moving from origin to left
    - Condition 2: Top half of circle centered at (1,0), moving from origin to right
    - Condition 3: Bottom half of circle centered at (1,0), moving from origin to right

    Args:
        num_points_per_condition (int): Number of data points per condition per time step.
        num_time_points (int): Number of discrete time steps.
        radius (float): Mean radius of the semicircles.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the radial distribution.
        device (torch.device): Device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (num_time_points, num_points_per_condition * 4, 3).
    """
    all_data_t = []
    
    for t in range(num_time_points):
        data_t_conds = []
        
        # Calculate progress from 0 to 1
        progress = t / (num_time_points - 1) if num_time_points > 1 else 0
        
        for c in range(4):
            # Determine the target angle based on condition and progress
            if c == 0:  # Top half of left circle (moving from 0 to π)
                target_angle = progress * np.pi  # 0 to π
                center_x = -1.0
            elif c == 1:  # Bottom half of left circle (moving from 0 to -π)
                target_angle = -progress * np.pi  # 0 to -π
                center_x = -1.0
            elif c == 2:  # Top half of right circle (moving from π to 0)
                target_angle = np.pi - progress * np.pi  # π to 0
                center_x = 1.0
            else:  # c == 3, Bottom half of right circle (moving from -π to 0)
                target_angle = -np.pi + progress * np.pi  # -π to 0
                center_x = 1.0
            
            # Generate warped normal distribution
            sampled_angles = vonmises.rvs(
                loc=target_angle,
                kappa=angular_concentration,
                size=num_points_per_condition
            )
            
            log_normal_mu = np.log(radius)
            log_normal_sigma = radial_std_dev
            sampled_radii = lognorm.rvs(
                s=log_normal_sigma,
                loc=0,
                scale=np.exp(log_normal_mu),
                size=num_points_per_condition
            )
            
            # Convert to Cartesian coordinates
            x = center_x + sampled_radii * np.cos(sampled_angles)
            y = sampled_radii * np.sin(sampled_angles)
            
            # Stack coordinates and add condition labels
            points_t = np.stack((x, y), axis=-1)
            points_tensor = torch.tensor(points_t, dtype=torch.float32, device=device)
            labels_tensor = torch.full((num_points_per_condition, 1), c, device=device)
            
            data_t_conds.append(torch.cat((points_tensor, labels_tensor), dim=1))
        
        all_data_t.append(torch.cat(data_t_conds, dim=0))
    
    return torch.stack(all_data_t, dim=0)

def generate_conditional_semicircle_marginal(num_points_per_condition: int,
                                            time: float,
                                            radius: float = 1.0,
                                            angular_concentration: float = 15.0,
                                            radial_std_dev: float = 0.1,
                                            device: torch.device = DEVICE):
    """
    Generate a marginal distribution for all 4 semicircle conditions at a single continuous time.

    Args:
        num_points_per_condition (int): Number of samples per condition.
        time (float): Continuous time input (normalized between 0 and 1).
        radius (float): Radius of the semicircles.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the radial distribution.
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: Data tensor of shape (4 * num_points_per_condition, 3),
                      where the last dimension is [x, y, condition].
    """
    data_t_conds = []
    
    # Process each condition
    for c in range(4):
        # Determine the target angle based on condition and time
        if c == 0:  # Top half of left circle (moving from 0 to π)
            target_angle = time * np.pi  # 0 to π
            center_x = -1.0
        elif c == 1:  # Bottom half of left circle (moving from 0 to -π)
            target_angle = -time * np.pi  # 0 to -π
            center_x = -1.0
        elif c == 2:  # Top half of right circle (moving from π to 0)
            target_angle = np.pi - time * np.pi  # π to 0
            center_x = 1.0
        else:  # c == 3, Bottom half of right circle (moving from -π to 0)
            target_angle = -np.pi + time * np.pi  # -π to 0
            center_x = 1.0
        
        # Generate warped normal distribution
        sampled_angles = vonmises.rvs(
            loc=target_angle,
            kappa=angular_concentration,
            size=num_points_per_condition
        )
        
        log_normal_mu = np.log(radius)
        log_normal_sigma = radial_std_dev
        sampled_radii = lognorm.rvs(
            s=log_normal_sigma,
            loc=0,
            scale=np.exp(log_normal_mu),
            size=num_points_per_condition
        )
        
        # Convert to Cartesian coordinates
        x = center_x + sampled_radii * np.cos(sampled_angles)
        y = sampled_radii * np.sin(sampled_angles)
        
        # Stack coordinates and add condition labels
        points = np.stack((x, y), axis=-1)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        labels_tensor = torch.full((num_points_per_condition, 1), c, device=device)
        
        data_t_conds.append(torch.cat((points_tensor, labels_tensor), dim=1))
    
    return torch.cat(data_t_conds, dim=0)



def log_likelihood_conditional_semicircle(
        data: torch.Tensor, # Shape (N, 3): [x, y, condition]
        time: float,
        radius: float = 1.0,
        angular_concentration: float = 5.0,
        radial_std_dev: float = 0.05 # Std dev for log(radius), 's' parameter for lognorm
    ) -> float:
    """
    Calculates the total log-likelihood of the given data under the semicircle model
    for a specific time and model parameters.

    Args:
        data (torch.Tensor): Input data of shape (N, 3), where columns are [x, y, condition].
                             Conditions should be integers 0, 1, 2, 3.
        time (float): Continuous time at which the data is assumed to be generated.
        radius (float): Radius of the semicircles.
        angular_concentration (float): Concentration parameter for the von Mises distribution.
        radial_std_dev (float): Standard deviation for the log of the radial distribution.

    Returns:
        float: The total log-likelihood of the dataset.
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Input data must have shape (N, 3)")

    data_np = data.cpu().numpy()
    x_coords = data_np[:, 0]
    y_coords = data_np[:, 1]
    conditions = data_np[:, 2].astype(int)

    total_log_likelihood = 0.0
    
    log_normal_mu_param = np.log(radius) # mu parameter for the underlying normal of lognorm

    for c_val in range(4):
        mask = (conditions == c_val)
        if not np.any(mask):
            continue

        current_x = x_coords[mask]
        current_y = y_coords[mask]

        # Determine target angle and center for the condition and time
        if c_val == 0:
            target_angle = time * np.pi
            center_x = -1.0
        elif c_val == 1:
            target_angle = -time * np.pi
            center_x = -1.0
        elif c_val == 2:
            target_angle = np.pi - time * np.pi
            center_x = 1.0
        else:  # c_val == 3
            target_angle = -np.pi + time * np.pi
            center_x = 1.0
        
        x_local = current_x - center_x
        y_local = current_y 

        epsilon = 1e-9 # To avoid log(0) or issues with r_polar=0
        r_polar = np.sqrt(x_local**2 + y_local**2)
        theta_polar = np.arctan2(y_local, x_local)

        log_pdf_angles = scipy_vonmises.logpdf(theta_polar, loc=target_angle, kappa=angular_concentration)
        valid_radii_mask = r_polar > epsilon
        log_pdf_radii = np.full_like(r_polar, -np.inf)

        if np.any(valid_radii_mask):
            log_pdf_radii[valid_radii_mask] = scipy_lognorm.logpdf(
                r_polar[valid_radii_mask],
                s=radial_std_dev,                # Shape parameter (sigma of underlying log)
                loc=0,                           # Shift parameter
                scale=np.exp(log_normal_mu_param) # Scale parameter (exp(mu of underlying log))
            )

        # Jacobian for polar to Cartesian transformation: pdf_cartesian = pdf_polar / r
        # So, log_pdf_cartesian = log_pdf_polar - log(r)
        log_jacobian_term = np.full_like(r_polar, -np.inf)
        if np.any(valid_radii_mask):
            log_jacobian_term[valid_radii_mask] = -np.log(r_polar[valid_radii_mask])

        # Sum log likelihoods for points in this condition (only for valid radii)
        current_log_likelihood = np.sum(
            log_pdf_angles[valid_radii_mask] + \
            log_pdf_radii[valid_radii_mask] + \
            log_jacobian_term[valid_radii_mask]
        )
        
        total_log_likelihood += current_log_likelihood
        
    return float(total_log_likelihood)

if __name__ == "__main__":
    #set seed
    torch.manual_seed(1)
    np.random.seed(1)
    
    if False:  # Eval semicircle marginals
        # Parameters
        num_points_per_condition = 100
        time = 0.5
        radius = 1.0
        angular_concentration = 5.0
        radial_std_dev = 0.05

        # Generate the marginal data for all 4 conditions at the selected time
        marginal_data_semicircle = generate_conditional_semicircle_marginal(
            num_points_per_condition=num_points_per_condition,
            time=time,
            radius=radius,
            angular_concentration=angular_concentration,
            radial_std_dev=radial_std_dev,
            device=DEVICE
        )

        print(f"Generated semicircle marginal data shape: {marginal_data_semicircle.shape}")

        x = marginal_data_semicircle[:, 0].cpu().numpy()
        y = marginal_data_semicircle[:, 1].cpu().numpy()
        conditions = marginal_data_semicircle[:, 2].cpu().numpy()

        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, 4))
        labels = ['Condition 0 (Left Top)', 'Condition 1 (Left Bottom)', 
                'Condition 2 (Right Top)', 'Condition 3 (Right Bottom)']

        for c in range(4):
            mask = conditions == c
            plt.scatter(x[mask], y[mask], label=labels[c], color=colors[c], alpha=0.6, s=10)

        from matplotlib.patches import Arc
        
        top_left_arc = Arc((-1, 0), 2*radius, 2*radius, 
                        theta1=0, theta2=180, 
                        color=colors[0], linestyle='-', linewidth=1.5)
        
        bottom_left_arc = Arc((-1, 0), 2*radius, 2*radius, 
                            theta1=180, theta2=360, 
                            color=colors[1], linestyle='-', linewidth=1.5)
        
        top_right_arc = Arc((1, 0), 2*radius, 2*radius, 
                        theta1=0, theta2=180, 
                        color=colors[2], linestyle='-', linewidth=1.5)
        
        bottom_right_arc = Arc((1, 0), 2*radius, 2*radius, 
                            theta1=180, theta2=360, 
                            color=colors[3], linestyle='-', linewidth=1.5)
        
        plt.gca().add_patch(top_left_arc)
        plt.gca().add_patch(bottom_left_arc)
        plt.gca().add_patch(top_right_arc)
        plt.gca().add_patch(bottom_right_arc)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Semicircle Marginal Distribution at Time {time}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-1.5, 1.5)

        plt.savefig(os.path.join(SCRIPT_PATH, 'plots', f'conditional_semicircles_marginal_time_{time}.png'))

        times = [0, 0.25, 0.75]

        time_samples_list_semicircle = []

        for t in times:
            samples = generate_conditional_semicircle_marginal(
                num_points_per_condition=num_points_per_condition,
                time=t,
                radius=radius,
                angular_concentration=angular_concentration,
                radial_std_dev=radial_std_dev,
                device=DEVICE
            )
            samples = jnp.array(samples.cpu().numpy())

            time_samples_list_semicircle.append((t, samples))

        for time, samples in time_samples_list_semicircle:
            print(f"Time: {time}, Samples Shape: {samples.shape}")

        with open(os.path.join(SCRIPT_PATH, 'eval_marginals_semicircle.pkl'), 'wb') as f:
            pickle.dump(time_samples_list_semicircle, f)
        
        print(f"Semicircle marginals saved to {os.path.join(SCRIPT_PATH, 'eval_marginals_semicircle.pkl')}")

    if True:  # Conditional semicircles
        #----------------------------------------------------#
        #  Conditional Semicircles Data                      #
        #----------------------------------------------------#

        print("\n--- Conditional Semicircles Data ---")
        num_points_per_condition = 100
        num_time_points = 3
        radius = 1.0
        angular_concentration = 5.0  
        radial_std_dev = 0.05  

        # Generate the conditional semicircles data
        conditional_semicircles_data = generate_conditional_semicircles(
            num_points_per_condition=num_points_per_condition,
            num_time_points=num_time_points,
            radius=radius,
            angular_concentration=angular_concentration,
            radial_std_dev=radial_std_dev,
            device=DEVICE
        )

        print(f"Generated conditional semicircles data shape: {conditional_semicircles_data.shape}")

        # Save the data
        save_path_semicircles = os.path.join(SCRIPT_PATH, 'data', 'conditional_semicircles.pt')
        torch.save(conditional_semicircles_data, save_path_semicircles)
        print(f"Conditional semicircles data saved to {save_path_semicircles}")