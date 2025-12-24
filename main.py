# 🏗️ PyTruss2D Pro - Enhanced Version
"""
🏗️ PyTruss2D Pro - Advanced 2D Truss Solver
Efficient, Error-Resistant with More Features
"""

import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum

# Initialize colorama
init(autoreset=True)

class SupportType(Enum):
    FIXED = "fixed"
    ROLLER = "roller"
    PINNED = "pinned"  # New feature

class MaterialType(Enum):
    STEEL = "steel"
    ALUMINUM = "aluminum"
    WOOD = "wood"
    CONCRETE = "concrete"
    CUSTOM = "custom"

@dataclass
class Material:
    name: str
    E: float  # Young's modulus (Pa)
    density: float  # kg/m³
    yield_strength: float  # Pa
    color: str

class Truss2DPro:
    """Enhanced 2D Truss Analysis with More Features"""
    
    # Predefined materials
    MATERIALS = {
        MaterialType.STEEL: Material("Steel", 200e9, 7850, 250e6, "#4682b4"),
        MaterialType.ALUMINUM: Material("Aluminum", 69e9, 2700, 110e6, "#87ceeb"),
        MaterialType.WOOD: Material("Wood", 10e9, 500, 30e6, "#8b4513"),
        MaterialType.CONCRETE: Material("Concrete", 25e9, 2400, 25e6, "#a9a9a9"),
    }
    
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.supports = {}
        self.loads = {}
        self.material = self.MATERIALS[MaterialType.STEEL]
        self.A = 0.01  # Cross-sectional area (m²)
        self.solution = None
        self.error_log = []
        
        # Auto-generated IDs
        self.node_id_counter = 0
        self.element_id_counter = 0
        
        # UI settings
        self.deformation_scale = 50
        self.show_grid = True
        self.auto_scale = True
        
    def reset(self):
        """Reset all data"""
        self.nodes.clear()
        self.elements.clear()
        self.supports.clear()
        self.loads.clear()
        self.solution = None
        self.error_log.clear()
        self.node_id_counter = 0
        self.element_id_counter = 0
    
    def add_node(self, x: float, y: float, name: str = None) -> int:
        """Add a node with automatic ID"""
        node_id = self.node_id_counter
        self.nodes.append({
            'id': node_id,
            'x': float(x),
            'y': float(y),
            'name': name or f"N{node_id}"
        })
        self.node_id_counter += 1
        return node_id
    
    def add_node_by_delta(self, dx: float, dy: float, from_node: int = None) -> int:
        """Add node relative to existing node"""
        if from_node is None and self.nodes:
            from_node = len(self.nodes) - 1
        
        if from_node is not None and 0 <= from_node < len(self.nodes):
            base = self.nodes[from_node]
            return self.add_node(base['x'] + dx, base['y'] + dy)
        else:
            return self.add_node(dx, dy)
    
    def add_element(self, n1: int, n2: int, name: str = None) -> int:
        """Add element with validation"""
        # Validate node indices
        if not self._validate_nodes([n1, n2]):
            self.log_error(f"Invalid node indices: {n1}, {n2}")
            return -1
        
        # Check for duplicate element
        for elem in self.elements:
            if (elem['nodes'][0] == n1 and elem['nodes'][1] == n2) or \
               (elem['nodes'][0] == n2 and elem['nodes'][1] == n1):
                self.log_error(f"Element already exists between nodes {n1} and {n2}")
                return elem['id']
        
        # Check for zero-length element
        x1, y1 = self.nodes[n1]['x'], self.nodes[n1]['y']
        x2, y2 = self.nodes[n2]['x'], self.nodes[n2]['y']
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < 0.001:
            self.log_error(f"Element too short ({length:.3f} m) between nodes {n1} and {n2}")
            return -1
        
        elem_id = self.element_id_counter
        self.elements.append({
            'id': elem_id,
            'nodes': (n1, n2),
            'name': name or f"E{elem_id}",
            'length': length
        })
        self.element_id_counter += 1
        return elem_id
    
    def add_rectangle(self, x: float, y: float, width: float, height: float) -> List[int]:
        """Add rectangular grid of nodes and elements"""
        node_ids = []
        
        # Add 4 corner nodes
        node_ids.append(self.add_node(x, y))
        node_ids.append(self.add_node(x + width, y))
        node_ids.append(self.add_node(x + width, y + height))
        node_ids.append(self.add_node(x, y + height))
        
        # Add perimeter elements
        self.add_element(node_ids[0], node_ids[1])
        self.add_element(node_ids[1], node_ids[2])
        self.add_element(node_ids[2], node_ids[3])
        self.add_element(node_ids[3], node_ids[0])
        
        # Add diagonal for stability
        self.add_element(node_ids[0], node_ids[2])
        
        return node_ids
    
    def add_support(self, node: int, support_type: SupportType):
        """Add support with validation"""
        if not self._validate_nodes([node]):
            self.log_error(f"Invalid node index for support: {node}")
            return
        
        self.supports[node] = support_type.value
    
    def add_load(self, node: int, fx: float = 0, fy: float = 0):
        """Add load with validation"""
        if not self._validate_nodes([node]):
            self.log_error(f"Invalid node index for load: {node}")
            return
        
        # Convert to kN if too large
        if abs(fx) > 10000 or abs(fy) > 10000:
            print(Fore.YELLOW + f"Note: Force values seem high. Make sure units are Newtons.")
        
        self.loads[node] = {'fx': float(fx), 'fy': float(fy)}
    
    def add_uniform_load(self, nodes: List[int], fy_per_node: float):
        """Add uniform load across multiple nodes"""
        for node in nodes:
            self.add_load(node, 0, fy_per_node)
    
    def set_material(self, material_type: MaterialType, A: float = None):
        """Set material properties"""
        if material_type in self.MATERIALS:
            self.material = self.MATERIALS[material_type]
            if A is not None:
                self.A = A
        else:
            self.log_error(f"Unknown material type: {material_type}")
    
    def set_custom_material(self, name: str, E: float, density: float, yield_strength: float):
        """Set custom material properties"""
        self.material = Material(name, E, density, yield_strength, "#555555")
    
    def _validate_nodes(self, node_indices: List[int]) -> bool:
        """Validate node indices exist"""
        for node in node_indices:
            if node < 0 or node >= len(self.nodes):
                return False
        return True
    
    def validate_model(self) -> Tuple[bool, str]:
        """Validate the entire model before solving"""
        errors = []
        
        # Check minimum requirements
        if len(self.nodes) < 2:
            errors.append("Need at least 2 nodes")
        
        if len(self.elements) < 1:
            errors.append("Need at least 1 element")
        
        if len(self.supports) < 2:
            errors.append("Need at least 2 supports for stability")
        
        # Check for unconnected nodes
        connected_nodes = set()
        for elem in self.elements:
            connected_nodes.add(elem['nodes'][0])
            connected_nodes.add(elem['nodes'][1])
        
        unconnected = [i for i in range(len(self.nodes)) if i not in connected_nodes]
        if unconnected:
            errors.append(f"Unconnected nodes: {unconnected}")
        
        # Check for mechanisms (quick stability check)
        n_dof = 2 * len(self.nodes)
        n_constraints = sum(2 if sup == 'fixed' else 1 for sup in self.supports.values())
        
        if n_dof - n_constraints > 0:
            errors.append(f"Structure may be unstable. DOF: {n_dof}, Constraints: {n_constraints}")
        
        # Check material properties
        if self.material.E <= 0:
            errors.append("Young's modulus must be positive")
        
        if self.A <= 0:
            errors.append("Cross-sectional area must be positive")
        
        if errors:
            return False, " | ".join(errors)
        return True, "Model is valid"
    
    def log_error(self, message: str):
        """Log error without crashing"""
        self.error_log.append(message)
        print(Fore.RED + f"⚠️ {message}")
    
    def print_errors(self):
        """Print all logged errors"""
        if self.error_log:
            print(Fore.RED + "\n" + "="*60)
            print(Fore.RED + "ERROR LOG")
            print(Fore.RED + "="*60)
            for i, error in enumerate(self.error_log, 1):
                print(Fore.RED + f"{i}. {error}")
            print()
    
    def solve(self, verbose: bool = True) -> bool:
        """Solve the truss system with error handling"""
        try:
            # Clear previous errors
            self.error_log.clear()
            
            # Validate model
            is_valid, message = self.validate_model()
            if not is_valid:
                self.log_error(f"Model validation failed: {message}")
                return False
            
            if verbose:
                print(Fore.CYAN + "\n" + "="*60)
                print(Fore.YELLOW + "SOLVING TRUSS SYSTEM")
                print(Fore.CYAN + "="*60)
                print(Fore.GREEN + f"Material: {self.material.name}")
                print(Fore.GREEN + f"Cross-section: {self.A:.6f} m²")
            
            # Get dimensions
            n_nodes = len(self.nodes)
            n_dof = 2 * n_nodes
            
            # Initialize matrices
            K = np.zeros((n_dof, n_dof))
            F = np.zeros(n_dof)
            
            # Assemble global stiffness matrix
            if verbose:
                print("Step 1: Assembling stiffness matrix...")
            
            for elem in self.elements:
                n1, n2 = elem['nodes']
                x1, y1 = self.nodes[n1]['x'], self.nodes[n1]['y']
                x2, y2 = self.nodes[n2]['x'], self.nodes[n2]['y']
                
                L = elem['length']
                c = (x2 - x1) / L
                s = (y2 - y1) / L
                
                # Element stiffness
                k_local = (self.material.E * self.A / L) * np.array([
                    [1, 0, -1, 0],
                    [0, 0, 0, 0],
                    [-1, 0, 1, 0],
                    [0, 0, 0, 0]
                ])
                
                # Transformation
                T = np.array([[c, s, 0, 0], [-s, c, 0, 0],
                            [0, 0, c, s], [0, 0, -s, c]])
                k_global = T.T @ k_local @ T
                
                # Assemble
                dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
                for i in range(4):
                    for j in range(4):
                        K[dofs[i], dofs[j]] += k_global[i, j]
            
            # Assemble force vector
            if verbose:
                print("Step 2: Assembling force vector...")
            
            for node, load in self.loads.items():
                F[2*node] += load['fx']
                F[2*node+1] += load['fy']
            
            # Apply boundary conditions
            if verbose:
                print("Step 3: Applying boundary conditions...")
            
            fixed_dofs = []
            for node, support_type in self.supports.items():
                if support_type == 'fixed':
                    fixed_dofs.extend([2*node, 2*node+1])
                else:  # roller or pinned
                    fixed_dofs.append(2*node+1)  # vertical constraint
            
            # Modify system for constraints
            K_mod = K.copy()
            F_mod = F.copy()
            for dof in fixed_dofs:
                K_mod[dof, :] = 0
                K_mod[:, dof] = 0
                K_mod[dof, dof] = 1
                F_mod[dof] = 0
            
            # Solve system
            if verbose:
                print("Step 4: Solving linear system...")
            
            try:
                displacements = np.linalg.solve(K_mod, F_mod)
            except np.linalg.LinAlgError:
                # Try least squares if direct solve fails
                displacements = np.linalg.lstsq(K_mod, F_mod, rcond=None)[0]
                self.log_error("Used least squares solution (direct solve failed)")
            
            # Calculate reactions
            reactions = K @ displacements
            
            # Calculate internal forces
            forces = []
            stresses = []
            for elem in self.elements:
                n1, n2 = elem['nodes']
                L = elem['length']
                x1, y1 = self.nodes[n1]['x'], self.nodes[n1]['y']
                x2, y2 = self.nodes[n2]['x'], self.nodes[n2]['y']
                
                c = (x2 - x1) / L
                s = (y2 - y1) / L
                
                # Local displacements
                u1_local = c*displacements[2*n1] + s*displacements[2*n1+1]
                u2_local = c*displacements[2*n2] + s*displacements[2*n2+1]
                
                # Axial force
                force = (self.material.E * self.A / L) * (u2_local - u1_local)
                stress = force / self.A
                
                forces.append(force)
                stresses.append(stress)
            
            # Store solution
            self.solution = {
                'displacements': displacements,
                'reactions': reactions,
                'forces': forces,
                'stresses': stresses,
                'fixed_dofs': fixed_dofs,
                'K_global': K
            }
            
            if verbose:
                print(Fore.GREEN + "\n✓ Analysis completed successfully!")
            
            return True
            
        except Exception as e:
            self.log_error(f"Solution failed: {str(e)}")
            return False
    
    def print_results(self, show_details: bool = True):
        """Print analysis results with formatting"""
        if not self.solution:
            print(Fore.RED + "No solution available. Run solve() first.")
            return
        
        print(Fore.CYAN + "\n" + "="*60)
        print(Fore.YELLOW + "ANALYSIS RESULTS")
        print(Fore.CYAN + "="*60)
        
        # Summary
        print(Fore.WHITE + f"\n📊 Summary:")
        print(f"  Material: {self.material.name}")
        print(f"  Young's Modulus: {self.material.E:.2e} Pa")
        print(f"  Cross-sectional Area: {self.A:.6f} m²")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Elements: {len(self.elements)}")
        print(f"  Supports: {len(self.supports)}")
        print(f"  Loads: {len(self.loads)}")
        
        # Quick stats
        max_disp = np.max(np.abs(self.solution['displacements']))
        max_force = np.max(np.abs(self.solution['forces']))
        max_stress = np.max(np.abs(self.solution['stresses']))
        
        print(Fore.WHITE + f"\n📈 Quick Stats:")
        print(f"  Max Displacement: {max_disp:.6f} m")
        print(f"  Max Internal Force: {max_force:.2f} N")
        print(f"  Max Stress: {max_stress/1e6:.2f} MPa")
        
        # Safety check
        safety_factors = []
        for stress in self.solution['stresses']:
            if abs(stress) > 1e-6:
                sf = self.material.yield_strength / abs(stress)
                safety_factors.append(sf)
        
        if safety_factors:
            min_sf = min(safety_factors)
            print(Fore.WHITE + f"\n🛡️ Safety Check:")
            print(f"  Yield Strength: {self.material.yield_strength/1e6:.0f} MPa")
            print(f"  Min Safety Factor: {min_sf:.2f}")
            if min_sf >= 2.0:
                print(Fore.GREEN + f"  Status: SAFE ✓")
            elif min_sf >= 1.5:
                print(Fore.YELLOW + f"  Status: MARGINAL ⚠")
            else:
                print(Fore.RED + f"  Status: UNSAFE ✗")
        
        if show_details:
            # Nodal displacements
            print(Fore.WHITE + "\n📍 Nodal Displacements:")
            print("Node\tu (mm)\t\tv (mm)\t\tTotal (mm)")
            print("-"*50)
            for i, node in enumerate(self.nodes):
                u = self.solution['displacements'][2*i] * 1000
                v = self.solution['displacements'][2*i+1] * 1000
                total = np.sqrt(u**2 + v**2)
                print(f"{i}\t{u:.3f}\t\t{v:.3f}\t\t{total:.3f}")
            
            # Element forces
            print(Fore.WHITE + "\n🔩 Element Forces:")
            print("Element\tForce (kN)\tStress (MPa)\tStatus")
            print("-"*50)
            for i, (force, stress) in enumerate(zip(self.solution['forces'], self.solution['stresses'])):
                force_kN = force / 1000
                stress_MPa = stress / 1e6
                
                if force > 0:
                    status = "TENSION"
                    color = Fore.RED
                elif force < 0:
                    status = "COMPRESSION"
                    color = Fore.BLUE
                else:
                    status = "ZERO"
                    color = Fore.WHITE
                
                print(f"{i}\t{force_kN:+.3f}\t\t{stress_MPa:+.3f}\t\t{color}{status}")
    
    def plot(self, show_deformed: bool = True, show_forces: bool = True, 
             show_reactions: bool = True, save_path: str = None):
        """Enhanced plotting with more options"""
        if not self.solution:
            print(Fore.RED + "No solution to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Plot undeformed elements
        for elem in self.elements:
            n1, n2 = elem['nodes']
            x1, y1 = self.nodes[n1]['x'], self.nodes[n1]['y']
            x2, y2 = self.nodes[n2]['x'], self.nodes[n2]['y']
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1.5, alpha=0.5, zorder=1)
        
        # Plot deformed shape
        if show_deformed:
            for elem in self.elements:
                n1, n2 = elem['nodes']
                x1, y1 = self.nodes[n1]['x'], self.nodes[n1]['y']
                x2, y2 = self.nodes[n2]['x'], self.nodes[n2]['y']
                
                u1 = self.solution['displacements'][2*n1] * self.deformation_scale
                v1 = self.solution['displacements'][2*n1+1] * self.deformation_scale
                u2 = self.solution['displacements'][2*n2] * self.deformation_scale
                v2 = self.solution['displacements'][2*n2+1] * self.deformation_scale
                
                ax.plot([x1+u1, x2+u2], [y1+v1, y2+v2], 'r--', 
                       linewidth=1.2, alpha=0.7, zorder=2, label='Deformed')
        
        # Plot elements colored by force
        if show_forces:
            max_force = max(abs(f) for f in self.solution['forces'])
            if max_force > 0:
                for i, elem in enumerate(self.elements):
                    n1, n2 = elem['nodes']
                    x1, y1 = self.nodes[n1]['x'], self.nodes[n1]['y']
                    x2, y2 = self.nodes[n2]['x'], self.nodes[n2]['y']
                    
                    force = self.solution['forces'][i]
                    
                    # Normalize force for coloring (-1 to 1)
                    norm_force = force / max_force
                    
                    # Choose color: red for tension, blue for compression
                    if norm_force > 0:
                        color = plt.cm.Reds(0.5 + norm_force/2)
                    else:
                        color = plt.cm.Blues(0.5 - norm_force/2)
                    
                    # Plot with thickness proportional to force magnitude
                    linewidth = 1 + 3 * abs(norm_force)
                    ax.plot([x1, x2], [y1, y2], color=color, 
                           linewidth=linewidth, solid_capstyle='round', zorder=3)
        
        # Plot nodes
        for i, node in enumerate(self.nodes):
            ax.plot(node['x'], node['y'], 'ko', markersize=8, zorder=4)
            ax.text(node['x'], node['y']+0.1, f"{i}", 
                   fontsize=9, ha='center', va='bottom', zorder=5,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot supports
        for node, support_type in self.supports.items():
            x, y = self.nodes[node]['x'], self.nodes[node]['y']
            
            if support_type == 'fixed':
                # Triangle
                triangle = plt.Polygon([(x-0.4, y-0.4), (x, y), (x+0.4, y-0.4)], 
                                     color='green', alpha=0.7)
                ax.add_patch(triangle)
                ax.text(x, y-0.6, 'Fixed', ha='center', va='top', fontsize=8)
            else:
                # Roller
                circle = plt.Circle((x, y-0.2), 0.2, color='orange', alpha=0.7)
                ax.add_patch(circle)
                ax.plot([x-0.4, x+0.4], [y-0.4, y-0.4], 'orange', linewidth=2)
                ax.text(x, y-0.7, 'Roller', ha='center', va='top', fontsize=8)
        
        # Plot loads
        for node, load in self.loads.items():
            x, y = self.nodes[node]['x'], self.nodes[node]['y']
            fx, fy = load['fx'], load['fy']
            
            if abs(fx) > 1 or abs(fy) > 1:
                scale = 0.0005
                ax.arrow(x, y, fx*scale, fy*scale, 
                        head_width=0.3, head_length=0.4, 
                        fc='red', ec='red', linewidth=2, zorder=6)
                
                # Label
                magnitude = np.sqrt(fx**2 + fy**2)
                if magnitude > 0:
                    ax.text(x + fx*scale, y + fy*scale, 
                           f'{magnitude/1000:.1f} kN', fontsize=8,
                           ha='center', va='bottom', zorder=7,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot reactions
        if show_reactions:
            for node in self.supports:
                x, y = self.nodes[node]['x'], self.nodes[node]['y']
                Rx = self.solution['reactions'][2*node]
                Ry = self.solution['reactions'][2*node+1]
                
                if abs(Rx) > 1 or abs(Ry) > 1:
                    scale = 0.0005
                    ax.arrow(x, y, -Rx*scale, -Ry*scale,
                            head_width=0.3, head_length=0.4,
                            fc='blue', ec='blue', linewidth=2, 
                            linestyle=':', zorder=6)
        
        # Formatting
        ax.set_aspect('equal')
        if self.show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X Position (m)', fontsize=11)
        ax.set_ylabel('Y Position (m)', fontsize=11)
        
        # Title with statistics
        if self.solution:
            max_disp = np.max(np.abs(self.solution['displacements'])) * 1000
            title = f"PyTruss2D Pro - {self.material.name} Truss\n"
            title += f"Max Displacement: {max_disp:.2f} mm | "
            title += f"Elements: {len(self.elements)}"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=1.5, label='Undeformed'),
            Line2D([0], [0], color='red', linewidth=1.2, linestyle='--', label='Deformed'),
            Line2D([0], [0], color='red', linewidth=2, label='Tension'),
            Line2D([0], [0], color='blue', linewidth=2, label='Compression'),
            Line2D([0], [0], color='green', linewidth=2, label='Fixed Support'),
            Line2D([0], [0], color='orange', linewidth=2, label='Roller Support'),
            Line2D([0], [0], color='red', linewidth=2, label='Load'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle=':', label='Reaction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Auto-scale
        if self.auto_scale:
            ax.autoscale_view()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(Fore.GREEN + f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def export_results(self, filename: str = "truss_results.txt"):
        """Export results to text file"""
        if not self.solution:
            print(Fore.RED + "No results to export")
            return
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PYTRUSS2D PRO - ANALYSIS RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Material: {self.material.name}\n")
            f.write(f"Young's Modulus: {self.material.E:.2e} Pa\n")
            f.write(f"Cross-sectional Area: {self.A:.6f} m²\n\n")
            
            f.write("Nodal Displacements (mm):\n")
            f.write("-"*40 + "\n")
            for i in range(len(self.nodes)):
                u = self.solution['displacements'][2*i] * 1000
                v = self.solution['displacements'][2*i+1] * 1000
                f.write(f"Node {i}: u={u:.3f}, v={v:.3f}\n")
            
            f.write("\nElement Forces (kN):\n")
            f.write("-"*40 + "\n")
            for i, force in enumerate(self.solution['forces']):
                force_kN = force / 1000
                f.write(f"Element {i}: {force_kN:+.3f} kN\n")
        
        print(Fore.GREEN + f"Results exported to {filename}")
    
    def create_report(self):
        """Create comprehensive report"""
        report = []
        report.append("="*60)
        report.append("PYTRUSS2D PRO - COMPREHENSIVE REPORT")
        report.append("="*60)
        
        # Model info
        report.append(f"\n📋 MODEL INFORMATION")
        report.append(f"   Material: {self.material.name}")
        report.append(f"   Cross-section: {self.A*10000:.1f} cm²")
        report.append(f"   Nodes: {len(self.nodes)}")
        report.append(f"   Elements: {len(self.elements)}")
        
        # Safety analysis
        if self.solution:
            max_stress = max(abs(s) for s in self.solution['stresses'])
            safety_factor = self.material.yield_strength / max_stress if max_stress > 0 else float('inf')
            
            report.append(f"\n🛡️ SAFETY ANALYSIS")
            report.append(f"   Max Stress: {max_stress/1e6:.2f} MPa")
            report.append(f"   Yield Strength: {self.material.yield_strength/1e6:.0f} MPa")
            report.append(f"   Safety Factor: {safety_factor:.2f}")
            
            if safety_factor >= 2.0:
                report.append(f"   Status: ✅ SAFE")
            elif safety_factor >= 1.5:
                report.append(f"   Status: ⚠ MARGINAL")
            else:
                report.append(f"   Status: ❌ UNSAFE")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        if self.solution:
            # Find critical element
            max_stress_idx = np.argmax([abs(s) for s in self.solution['stresses']])
            max_stress = self.solution['stresses'][max_stress_idx]
            
            if abs(max_stress) > self.material.yield_strength * 0.8:
                report.append(f"   1. Strengthen Element {max_stress_idx} (highest stress)")
            
            # Check for zero-force members
            zero_force_elements = [i for i, f in enumerate(self.solution['forces']) if abs(f) < 1]
            if zero_force_elements:
                report.append(f"   2. Consider removing elements {zero_force_elements} (near-zero force)")
        
        report.append(f"   3. Verify all connections are properly modeled")
        report.append(f"   4. Consider load combinations and factors of safety")
        
        return "\n".join(report)

# ============================================================================
# USER-FRIENDLY INTERFACE
# ============================================================================

def run_interactive():
    """User-friendly interactive interface"""
    print(Fore.CYAN + "="*60)
    print(Fore.YELLOW + "🏗️ PYTRUSS2D PRO - INTERACTIVE MODE")
    print(Fore.CYAN + "="*60)
    
    truss = Truss2DPro()
    
    while True:
        print(Fore.WHITE + "\n📋 MAIN MENU:")
        print("1. 📍 Add Nodes")
        print("2. 🔗 Add Elements")
        print("3. 🏗️ Add Supports")
        print("4. ⬇️ Add Loads")
        print("5. 🧱 Set Material")
        print("6. ⚡ Quick Templates")
        print("7. 🔍 Solve & Analyze")
        print("8. 📊 View Results")
        print("9. 🎨 Plot")
        print("10. 💾 Export")
        print("0. 🚪 Exit")
        
        choice = input(Fore.YELLOW + "\nEnter choice (0-10): " + Style.RESET_ALL)
        
        if choice == '1':
            print(Fore.WHITE + "\n📍 ADD NODES:")
            print("1. Single node (enter coordinates)")
            print("2. Multiple nodes (grid)")
            print("3. Rectangle")
            
            sub = input("Choose (1-3): ")
            
            if sub == '1':
                try:
                    x = float(input("X coordinate: "))
                    y = float(input("Y coordinate: "))
                    name = input("Node name (optional): ")
                    nid = truss.add_node(x, y, name)
                    print(Fore.GREEN + f"✓ Added node {nid} at ({x}, {y})")
                except:
                    print(Fore.RED + "Invalid input!")
            
            elif sub == '2':
                try:
                    x0 = float(input("Start X: "))
                    y0 = float(input("Start Y: "))
                    dx = float(input("Spacing X: "))
                    dy = float(input("Spacing Y: "))
                    nx = int(input("Number in X: "))
                    ny = int(input("Number in Y: "))
                    
                    for i in range(nx):
                        for j in range(ny):
                            x = x0 + i * dx
                            y = y0 + j * dy
                            truss.add_node(x, y)
                    print(Fore.GREEN + f"✓ Added {nx*ny} nodes in grid")
                except:
                    print(Fore.RED + "Invalid input!")
            
            elif sub == '3':
                try:
                    x = float(input("Bottom-left X: "))
                    y = float(input("Bottom-left Y: "))
                    w = float(input("Width: "))
                    h = float(input("Height: "))
                    nodes = truss.add_rectangle(x, y, w, h)
                    print(Fore.GREEN + f"✓ Added rectangle with nodes {nodes}")
                except:
                    print(Fore.RED + "Invalid input!")
        
        elif choice == '2':
            print(Fore.WHITE + "\n🔗 ADD ELEMENTS:")
            print("Available nodes:")
            for node in truss.nodes:
                print(f"  {node['id']}: ({node['x']}, {node['y']})")
            
            try:
                n1 = int(input("First node ID: "))
                n2 = int(input("Second node ID: "))
                eid = truss.add_element(n1, n2)
                if eid >= 0:
                    print(Fore.GREEN + f"✓ Added element {eid}")
                else:
                    print(Fore.RED + "Failed to add element")
            except:
                print(Fore.RED + "Invalid input!")
        
        elif choice == '3':
            print(Fore.WHITE + "\n🏗️ ADD SUPPORTS:")
            print("Available nodes:")
            for node in truss.nodes:
                print(f"  {node['id']}: ({node['x']}, {node['y']})")
            
            try:
                node = int(input("Node ID: "))
                print("Support types: fixed, roller")
                s_type = input("Type: ").lower()
                if s_type in ['fixed', 'roller']:
                    truss.add_support(node, SupportType(s_type))
                    print(Fore.GREEN + f"✓ Added {s_type} support at node {node}")
                else:
                    print(Fore.RED + "Invalid support type!")
            except:
                print(Fore.RED + "Invalid input!")
        
        elif choice == '4':
            print(Fore.WHITE + "\n⬇️ ADD LOADS:")
            print("Available nodes:")
            for node in truss.nodes:
                print(f"  {node['id']}: ({node['x']}, {node['y']})")
            
            try:
                node = int(input("Node ID: "))
                fx = float(input("Horizontal force (N): "))
                fy = float(input("Vertical force (N): "))
                truss.add_load(node, fx, fy)
                print(Fore.GREEN + f"✓ Added load ({fx}, {fy}) N at node {node}")
            except:
                print(Fore.RED + "Invalid input!")
        
        elif choice == '5':
            print(Fore.WHITE + "\n🧱 SET MATERIAL:")
            print("1. Steel (default)")
            print("2. Aluminum")
            print("3. Wood")
            print("4. Concrete")
            print("5. Custom")
            
            try:
                sub = input("Choose (1-5): ")
                if sub == '1':
                    truss.set_material(MaterialType.STEEL)
                    print(Fore.GREEN + "✓ Set material: Steel")
                elif sub == '2':
                    truss.set_material(MaterialType.ALUMINUM)
                    print(Fore.GREEN + "✓ Set material: Aluminum")
                elif sub == '3':
                    truss.set_material(MaterialType.WOOD)
                    print(Fore.GREEN + "✓ Set material: Wood")
                elif sub == '4':
                    truss.set_material(MaterialType.CONCRETE)
                    print(Fore.GREEN + "✓ Set material: Concrete")
                elif sub == '5':
                    name = input("Material name: ")
                    E = float(input("Young's modulus (Pa): "))
                    density = float(input("Density (kg/m³): "))
                    strength = float(input("Yield strength (Pa): "))
                    truss.set_custom_material(name, E, density, strength)
                    print(Fore.GREEN + f"✓ Set custom material: {name}")
            except:
                print(Fore.RED + "Invalid input!")
        
        elif choice == '6':
            print(Fore.WHITE + "\n⚡ QUICK TEMPLATES:")
            print("1. Simple Bridge")
            print("2. Roof Truss")
            print("3. Cantilever")
            print("4. Tower")
            
            try:
                sub = input("Choose (1-4): ")
                truss.reset()
                
                if sub == '1':
                    # Simple bridge
                    truss.add_node(0, 0)    # 0
                    truss.add_node(3, 0)    # 1
                    truss.add_node(6, 0)    # 2
                    truss.add_node(1.5, 2)  # 3
                    truss.add_node(4.5, 2)  # 4
                    
                    truss.add_element(0, 1)
                    truss.add_element(1, 2)
                    truss.add_element(0, 3)
                    truss.add_element(1, 3)
                    truss.add_element(1, 4)
                    truss.add_element(2, 4)
                    truss.add_element(3, 4)
                    
                    truss.add_support(0, SupportType.FIXED)
                    truss.add_support(2, SupportType.ROLLER)
                    
                    truss.add_load(3, 0, -10000)
                    truss.add_load(4, 0, -8000)
                    
                    print(Fore.GREEN + "✓ Created simple bridge template")
                
                elif sub == '2':
                    # Roof truss
                    truss.add_node(0, 0)    # 0
                    truss.add_node(4, 0)    # 1
                    truss.add_node(8, 0)    # 2
                    truss.add_node(2, 2)    # 3
                    truss.add_node(6, 2)    # 4
                    truss.add_node(4, 3)    # 5
                    
                    truss.add_element(0, 1)
                    truss.add_element(1, 2)
                    truss.add_element(0, 3)
                    truss.add_element(1, 3)
                    truss.add_element(1, 4)
                    truss.add_element(2, 4)
                    truss.add_element(3, 4)
                    truss.add_element(3, 5)
                    truss.add_element(4, 5)
                    
                    truss.add_support(0, SupportType.FIXED)
                    truss.add_support(2, SupportType.ROLLER)
                    
                    truss.add_uniform_load([3, 4, 5], -5000)
                    
                    print(Fore.GREEN + "✓ Created roof truss template")
                
                elif sub == '3':
                    # Cantilever
                    for i in range(5):
                        truss.add_node(i*2, 0)
                        truss.add_node(i*2+1, 1)
                    
                    # Add elements (simplified)
                    for i in range(4):
                        truss.add_element(i*2, (i+1)*2)
                        truss.add_element(i*2, i*2+1)
                        truss.add_element(i*2+1, (i+1)*2)
                    
                    truss.add_support(0, SupportType.FIXED)
                    truss.add_support(1, SupportType.FIXED)
                    
                    truss.add_load(8, 0, -15000)
                    
                    print(Fore.GREEN + "✓ Created cantilever template")
                
                elif sub == '4':
                    # Tower
                    truss.add_node(0, 0)    # 0
                    truss.add_node(3, 0)    # 1
                    truss.add_node(0, 3)    # 2
                    truss.add_node(3, 3)    # 3
                    truss.add_node(0, 6)    # 4
                    truss.add_node(3, 6)    # 5
                    
                    # Vertical elements
                    truss.add_element(0, 2)
                    truss.add_element(1, 3)
                    truss.add_element(2, 4)
                    truss.add_element(3, 5)
                    
                    # Horizontal elements
                    truss.add_element(0, 1)
                    truss.add_element(2, 3)
                    truss.add_element(4, 5)
                    
                    # Diagonal elements
                    truss.add_element(0, 3)
                    truss.add_element(1, 2)
                    truss.add_element(2, 5)
                    truss.add_element(3, 4)
                    
                    truss.add_support(0, SupportType.FIXED)
                    truss.add_support(1, SupportType.FIXED)
                    
                    truss.add_load(4, 0, -10000)
                    truss.add_load(5, 0, -10000)
                    
                    print(Fore.GREEN + "✓ Created tower template")
            
            except:
                print(Fore.RED + "Invalid input!")
        
        elif choice == '7':
            print(Fore.WHITE + "\n🔍 SOLVING...")
            success = truss.solve(verbose=True)
            if success:
                print(Fore.GREEN + "\n✓ Analysis complete!")
                # Show report
                report = truss.create_report()
                print(Fore.CYAN + "\n" + report)
            else:
                print(Fore.RED + "\n✗ Analysis failed!")
                truss.print_errors()
        
        elif choice == '8':
            if truss.solution:
                truss.print_results(show_details=True)
            else:
                print(Fore.YELLOW + "No results available. Run analysis first.")
        
        elif choice == '9':
            if truss.nodes:
                print(Fore.WHITE + "\n🎨 PLOT OPTIONS:")
                print("1. Basic plot")
                print("2. With deformation")
                print("3. With force colors")
                print("4. With reactions")
                print("5. All features")
                
                try:
                    sub = input("Choose (1-5): ")
                    if sub == '1':
                        truss.plot(show_deformed=False, show_forces=False, show_reactions=False)
                    elif sub == '2':
                        truss.plot(show_deformed=True, show_forces=False, show_reactions=False)
                    elif sub == '3':
                        truss.plot(show_deformed=True, show_forces=True, show_reactions=False)
                    elif sub == '4':
                        truss.plot(show_deformed=True, show_forces=True, show_reactions=True)
                    elif sub == '5':
                        truss.plot(show_deformed=True, show_forces=True, show_reactions=True)
                except:
                    print(Fore.RED + "Plotting failed!")
            else:
                print(Fore.YELLOW + "No truss to plot. Create one first.")
        
        elif choice == '10':
            if truss.solution:
                filename = input("Export filename (default: truss_results.txt): ") or "truss_results.txt"
                truss.export_results(filename)
            else:
                print(Fore.YELLOW + "No results to export.")
        
        elif choice == '0':
            print(Fore.GREEN + "\n👋 Thank you for using PyTruss2D Pro!")
            break
        
        else:
            print(Fore.RED + "Invalid choice!")

def quick_demo():
    """Quick demonstration"""
    print(Fore.CYAN + "="*60)
    print(Fore.YELLOW + "🏗️ PYTRUSS2D PRO - QUICK DEMO")
    print(Fore.CYAN + "="*60)
    
    truss = Truss2DPro()
    
    # Create a simple bridge
    print(Fore.WHITE + "\n1. Creating bridge truss...")
    truss.add_node(0, 0)    # Left support
    truss.add_node(3, 0)    # Middle left
    truss.add_node(6, 0)    # Right support
    truss.add_node(1.5, 2)  # Top left
    truss.add_node(4.5, 2)  # Top right
    
    print(Fore.WHITE + "2. Adding elements...")
    truss.add_element(0, 1)  # Bottom chord
    truss.add_element(1, 2)  # Bottom chord
    truss.add_element(0, 3)  # Diagonal
    truss.add_element(1, 3)  # Vertical
    truss.add_element(1, 4)  # Diagonal
    truss.add_element(2, 4)  # Vertical
    truss.add_element(3, 4)  # Top chord
    
    print(Fore.WHITE + "3. Adding supports...")
    truss.add_support(0, SupportType.FIXED)   # Left fixed
    truss.add_support(2, SupportType.ROLLER)  # Right roller
    
    print(Fore.WHITE + "4. Adding loads...")
    truss.add_load(3, 0, -15000)  # 15 kN downward
    truss.add_load(4, 5000, -5000)  # Diagonal load
    
    print(Fore.WHITE + "5. Setting material...")
    truss.set_material(MaterialType.STEEL, A=0.005)
    
    print(Fore.WHITE + "6. Solving...")
    if truss.solve(verbose=True):
        print(Fore.GREEN + "\n✓ Analysis successful!")
        
        # Show results
        truss.print_results(show_details=True)
        
        # Show report
        report = truss.create_report()
        print(Fore.CYAN + "\n" + report)
        
        # Plot
        print(Fore.WHITE + "\n7. Generating plot...")
        truss.plot(show_deformed=True, show_forces=True, show_reactions=True)
        
        # Export
        truss.export_results("demo_results.txt")
        print(Fore.GREEN + "\n✅ Demo completed!")
    else:
        truss.print_errors()

if __name__ == "__main__":
    print(Fore.CYAN + "="*60)
    print(Fore.YELLOW + "🏗️ PYTRUSS2D PRO - ENHANCED VERSION")
    print(Fore.CYAN + "="*60)
    print(Fore.GREEN + "Efficient 2D Truss Analysis with Error Handling")
    print(Fore.CYAN + "-"*60)
    
    print(Fore.WHITE + "\nSelect mode:")
    print("1. 🎮 Interactive Mode (Step-by-step)")
    print("2. ⚡ Quick Demo (See it in action)")
    print("3. 📝 Script Mode (Use in your code)")
    
    try:
        mode = input(Fore.YELLOW + "\nEnter choice (1-3): " + Style.RESET_ALL)
        
        if mode == '1':
            run_interactive()
        elif mode == '2':
            quick_demo()
        elif mode == '3':
            print(Fore.WHITE + "\n📝 Script Mode Example:")
            print("""
# In your Python code:
from PyTruss2D_Pro import Truss2DPro, MaterialType, SupportType

truss = Truss2DPro()
# Build your truss here
# Solve: truss.solve()
# Plot: truss.plot()
# Export: truss.export_results()
            """)
        else:
            print(Fore.RED + "Invalid choice! Running demo...")
            quick_demo()
    
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nProgram interrupted.")
    except Exception as e:
        print(Fore.RED + f"\nError: {str(e)}")