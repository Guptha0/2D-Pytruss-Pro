🏗️ PyTruss2D Pro - Enhanced 2D Truss Analysis Solver
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/FEA-Direct_Stiffness_Method-orange.svg
https://img.shields.io/badge/Status-Stable-brightgreen.svg

A professional-grade yet simple 2D truss analysis tool using Finite Element Analysis (FEA) with enhanced error handling, multiple materials, and beautiful visualizations.

📋 What It Does (In Simple Terms)
Imagine you're building a Lego bridge. This tool answers:

🎯 Will it bend? → Calculates deformations

🎯 Which pieces are stressed? → Shows red/blue colors

🎯 Will it break? → Checks safety factors

🎯 How strong must supports be? → Calculates reactions

Basically: It's a crystal ball for structures! 🔮

🚀 Quick Start
Installation (30 seconds)
bash
# 1. Save the code as PyTruss2D_Pro.py
# 2. Install dependencies (only once)
pip install numpy matplotlib colorama

# 3. Run it!
python PyTruss2D_Pro.py
First Run (Choose Option 2)
text
Select mode:
1. 🎮 Interactive Mode (Step-by-step)
2. ⚡ Quick Demo (See it in action)  ← PRESS 2!
3. 📝 Script Mode (Use in your code)
That's it! The demo will automatically:

Build a bridge

Add loads

Analyze everything

Show colorful results

Create a safety report

🎮 Interactive Mode (Easy!)
Step 1: Add Nodes
text
📍 ADD NODES:
1. Single node (enter coordinates)
2. Multiple nodes (grid)
3. Rectangle
Just use these easy coordinates:

text
Triangle: (0,0) (4,0) (2,3)
Bridge: (0,0) (3,0) (6,0) (1.5,2) (4.5,2)
Square: (0,0) (4,0) (4,4) (0,4)
Step 2: Connect Nodes
text
🔗 ADD ELEMENTS:
Available nodes:
  0: (0.0, 0.0)
  1: (4.0, 0.0)
Enter: 0 1  ← Connects node 0 to 1
Step 3: Add Supports
text
🏗️ ADD SUPPORTS:
Enter: 0 fixed   ← Left support
Enter: 2 roller  ← Right support
Step 4: Add Loads
text
⬇️ ADD LOADS:
Enter: 3 0 -10000  ← 10 kN down at node 3
Step 5: Solve!
text
🔍 SOLVING...
✓ Analysis complete!
📊 What You Get
1. Beautiful Color Plot
🔴 Red beams = Tension (being stretched)

🔵 Blue beams = Compression (being squashed)

🟢 Green triangles = Fixed supports

🟠 Orange circles = Roller supports

🟥 Red arrows = Applied loads

🔷 Blue dotted arrows = Support reactions

2. Smart Results Table
text
📍 Nodal Displacements:
Node    u (mm)      v (mm)      Total (mm)
0       0.000       0.000       0.000
1       1.234       -2.345      2.654

🔩 Element Forces:
Element Force (kN)   Stress (MPa)   Status
0       +12.345      +123.45       TENSION
1       -9.876       -98.76        COMPRESSION
3. Safety Check
text
🛡️ SAFETY ANALYSIS:
Max Stress: 123.45 MPa
Yield Strength: 250 MPa
Safety Factor: 2.02
Status: ✅ SAFE
4. Recommendations
text
💡 RECOMMENDATIONS:
1. Strengthen Element 0 (highest stress)
2. Verify all connections
3. Consider load combinations
🌟 Key Features
🛡️ Error Protection
Won't crash if you make mistakes

Tells you exactly what's wrong

Suggests how to fix it

Validates everything before solving

🎨 Multiple Materials
python
# Built-in materials:
truss.set_material(MaterialType.STEEL)      # 🏗️ Steel (default)
truss.set_material(MaterialType.ALUMINUM)   # ✈️ Aluminum
truss.set_material(MaterialType.WOOD)       # 🌲 Wood
truss.set_material(MaterialType.CONCRETE)   # 🏢 Concrete

# Or create custom:
truss.set_custom_material("Titanium", 110e9, 4500, 900e6)
⚡ Quick Templates
Press 6 in interactive mode for:

Simple Bridge - Basic truss bridge

Roof Truss - House roof structure

Cantilever - Overhanging structure

Tower - Vertical tower structure

📈 Enhanced Calculations
✅ Safety factors

✅ Stress in MPa (easy to read)

✅ Displacements in mm

✅ Force in kN

✅ Automatic unit conversion

💾 Export & Save
python
# Save results to file
truss.export_results("my_bridge_results.txt")

# Create comprehensive report
report = truss.create_report()
print(report)
🔧 Programmatic Usage
Simple Script
python
from PyTruss2D_Pro import Truss2DPro, MaterialType, SupportType

# Create truss
truss = Truss2DPro()

# Build a simple bridge
truss.add_node(0, 0)    # Left support
truss.add_node(4, 0)    # Middle
truss.add_node(8, 0)    # Right support
truss.add_node(2, 3)    # Top left
truss.add_node(6, 3)    # Top right

# Connect elements
truss.add_element(0, 1)  # Bottom
truss.add_element(1, 2)  # Bottom
truss.add_element(0, 3)  # Diagonal
truss.add_element(1, 3)  # Vertical
truss.add_element(1, 4)  # Diagonal
truss.add_element(2, 4)  # Vertical
truss.add_element(3, 4)  # Top

# Add supports
truss.add_support(0, SupportType.FIXED)
truss.add_support(2, SupportType.ROLLER)

# Add load
truss.add_load(3, 0, -10000)  # 10 kN down

# Set material
truss.set_material(MaterialType.STEEL, A=0.005)

# Solve
truss.solve()

# Get results
truss.print_results()
truss.plot()
Advanced Features
python
# Add multiple nodes at once
truss.add_rectangle(0, 0, 4, 3)  # Creates 4-node rectangle

# Add uniform load
truss.add_uniform_load([1, 2, 3], -5000)  # 5 kN on each node

# Check model validity
is_valid, message = truss.validate_model()
if is_valid:
    truss.solve()
else:
    print(f"Fix: {message}")

# View errors
truss.print_errors()
🎯 Real-World Applications
For Students 🎓
Homework verification

Project design help

Understanding FEA visually

Exam preparation

For Teachers 👨‍🏫
Classroom demonstrations

Create example problems

Visual teaching aid

Grading assistance

For DIY Projects 🔨
Deck design

Shelf brackets

Small bridge planning

Playhouse structures

For Professionals 🏢
Preliminary design checks

Client presentations

Quick feasibility studies

Training new engineers

📁 File Structure
text
PyTruss2D_Pro/
├── PyTruss2D_Pro.py          # Main program (all-in-one)
├── requirements.txt           # Dependencies
├── demo_results.txt          # Example output
├── truss_plot.png           # Example plot
└── README.md                # This file
🛠️ Technical Details
Mathematical Foundation
The solver uses the Direct Stiffness Method:

text
{F} = [K]{u}
Where:

{F} = Force vector

[K] = Global stiffness matrix

{u} = Displacement vector

Element Types Supported
🔸 Truss elements (axial loads only)

🔸 2D analysis (planar structures)

🔸 Linear elastic (small deformations)

🔸 Pin-jointed (no bending moments)

Material Properties
Material	E (GPa)	Density (kg/m³)	Yield (MPa)	Color
Steel	200	7850	250	🔵
Aluminum	69	2700	110	⚪
Wood	10	500	30	🟤
Concrete	25	2400	25	🐘
🚨 Common Issues & Solutions
Problem	Solution
"Need at least 2 supports"	Add one fixed + one roller
"Element too short"	Increase distance between nodes
"Structure may be unstable"	Add more diagonal elements
"Invalid node indices"	Check node numbers (0, 1, 2...)
Plot doesn't show	Install matplotlib: pip install matplotlib
Colors don't appear	Enable colored output in terminal
📊 Sample Output
Console Output
text
============================================================
PYTRUSS2D PRO - ANALYSIS RESULTS
============================================================

📊 Summary:
  Material: Steel
  Young's Modulus: 2.00e+11 Pa
  Cross-sectional Area: 0.005000 m²
  Nodes: 5
  Elements: 7
  Supports: 2
  Loads: 2

📈 Quick Stats:
  Max Displacement: 5.432 mm
  Max Internal Force: 25.678 kN
  Max Stress: 128.39 MPa

🛡️ Safety Check:
  Yield Strength: 250 MPa
  Min Safety Factor: 1.95
  Status: MARGINAL ⚠
Plot Features
https://via.placeholder.com/800x400/1a237e/ffffff?text=Colorful+Truss+Visualization

🔮 Future Features (Planned)
3D truss analysis

Dynamic load analysis

Buckling check

Cost estimation

CAD file export

Mobile app version

Cloud sharing

🤝 Contributing
Want to improve PyTruss2D Pro?

Fork the repository

Create a feature branch

Test your changes

Submit a pull request

Feature Ideas:
Add more material types

Improve visualization

Add example library

Create GUI interface

Add load combinations

📚 Learning Resources
For Beginners
Start with Quick Demo

Try Interactive Mode

Use Quick Templates

Modify existing examples

For Intermediate Users
Study the script examples

Experiment with different materials

Try complex structures

Compare with hand calculations

For Advanced Users
Modify solver algorithms

Add new element types

Implement optimization

Extend to frame analysis

🆘 Getting Help
Quick Fixes:
Install issues: pip install --upgrade numpy matplotlib colorama

Plot issues: Make sure matplotlib is installed

Crash on start: Check Python version (needs 3.8+)

No colors: Terminal might not support colors

Still Stuck?
Run the demo first

Check error messages

Simplify your structure

Use quick templates

📄 License
MIT License - Free for educational, personal, and commercial use.

🙏 Acknowledgments
Finite Element Method pioneers

Python scientific computing community

Engineering educators worldwide

Open source contributors

📞 Contact & Support
Need help? Have ideas?

Create a GitHub issue

Email for support

Join discussions

🎯 Final Tip
Start with this simple test:

bash
python PyTruss2D_Pro.py
2  # Choose demo
Watch the magic happen in 30 seconds! 🎉

Built with ❤️ for engineers, students, and makers everywhere! 🏗️✨

"The best way to predict the future is to create it." - Peter Drucker# 2D-Pytruss-Pro
