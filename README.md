# WarpVBD

Implementing VBD in NVIDIA Warp. Tetrahedral meshes supported with both St. Venant and Stable Neo-hookean material models.

## Installation

Just PBRT-v3 needs to be built for visualization purposes. Run `./install.sh` for this build with CMake.

For the Python packaging only `warp-lang`, `numpy`, and `matplotlib` are required. We have a simple conda environment provided:
```
conda env create -f environment.yml
conda activate warp_vbd
```

## Demos

We present several examples that show the usecases of the simulator.

### Cantilever

A classical example for soft body simulation is a beam, fixed on one side, falling under gravity. We show the displacement of the free tip face of the beam on the right. This example can be reproduced by running:
```
python run_cantilever.py --render
```

<p align="center" style="display: flex; justify-content: center; gap: 20px;">
    <video height="300px" controls>
        <source src="asset/imgs/cantilever.mp4" type="video/mp4">
    </video>
    <img src="asset/imgs/displacement_cantilever.png" height="300px"/>
</p>


### Mass Spring System

Another standard soft body system is a mass falling under gravity attached through a thin string spring. We approximate the whole structure with voxels, which are converted to tetrahedral elements, where the mass' edges are by default 11 voxels and the spring is 1 voxel. The top of the spring is fixed in space. We show the displacement of the free bottom face of the mass on the right. This example can be reproduced by running:
```
python run_mass_spring.py --render
```

<p align="center" style="display: flex; justify-content: center; gap: 20px;">
    <video height="300px" controls>
        <source src="asset/imgs/mass_spring.mp4" type="video/mp4">
    </video>
    <img src="asset/imgs/displacement_mass_spring.png" height="300px"/>
</p>





## Notes

The compilation of the VBD solver kernel takes about 20s, so please be patient the first time the warp kernel is compiled.