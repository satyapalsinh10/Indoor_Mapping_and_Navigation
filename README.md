# Habitat-Sim

This project utilizes the powerful Habitat-Sim, a high-performance physics-enabled 3D simulator, to generate datasets for embodied AI tasks. The simulator supports 3D scans of indoor/outdoor spaces, CAD models, configurable sensors, and robots described via URDF, all while prioritizing simulation speed. The focus is on Habitat-Matterport 3D Research Dataset [HM3D](https://aihabitat.org/datasets/hm3d/) for dataset generation, specifically creating equirectangular videos from the agent's perspective navigating the environment.

The design philosophy of Habitat is to prioritize simulation speed over the breadth of simulation capabilities. When rendering a scene from the Matterport3D dataset, Habitat-Sim achieves several thousand frames per second (FPS) running single-threaded and reaches over 10,000 FPS multi-process on a single GPU.

Habitat-Sim is typically used with
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab), a modular high-level library for end-to-end experiments in embodied AI -- defining embodied AI tasks (e.g. navigation, instruction following, question answering), training agents (via imitation or reinforcement learning, or no learning at all as in classical SensePlanAct pipelines), and benchmarking their performance on the defined tasks using standard metrics.

## Features

- **Habitat-Sim Integration:** Leverage the capabilities of Habitat-Sim for high-speed, physics-enabled 3D simulation.
  
- **Dataset Support:** Utilize 3D scans, CAD models, and configurable sensors to create diverse datasets for embodied AI tasks.
  
- **HM3D Compatibility:** Seamlessly integrate with the Habitat-Matterport 3D Research Dataset for realistic environment representation.

- **Equirectangular Video Generation:** Record immersive equirectangular videos from the agent's camera during navigation.


## Workflow

- **Habitat-Sim Setup:** Configure and set up Habitat-Sim with the desired datasets, including HM3D.

- **Agent Navigation:** Simulate the agent navigating the environment, capturing equirectangular videos in the process.

- **Dataset Generation:** Use the recorded videos as input for dataset generation, focusing on embodied AI tasks.

- **Integration with Open V-SLAM:** Employ the generated datasets as input for the Open V-SLAM pipeline to create 3D sparse point clouds of the environment.


## Installation

Habitat-Sim can be installed in 3 ways:
1. Via Conda - Recommended method for most users. Stable release and nightly builds.
1. [Experimental] Via PIP - `pip install .` to compile the latest headless build with Bullet. Read [build instructions and common build issues](BUILD_FROM_SOURCE.md).
1. Via Docker - Updated approximately once per year for the [Habitat Challenge](https://aihabitat.org/challenge/). Read [habitat-docker-setup](https://github.com/facebookresearch/habitat-lab#docker-setup).
1. Via Source - For active development. Read [build instructions and common build issues](BUILD_FROM_SOURCE.md).

### [Recommended] Conda Packages

Habitat is under active development, and we advise users to restrict themselves to [stable releases](https://github.com/facebookresearch/habitat-sim/releases). Starting with v0.1.4, we provide [conda packages for each release](https://anaconda.org/aihabitat).

1. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
   ```bash
   # We require python>=3.9 and cmake>=3.10
   conda create -n habitat python=3.9 cmake=3.14.0
   conda activate habitat
   ```

1. **conda install habitat-sim**

   Pick one of the options below depending on your system/needs:

   - To install on machines with an attached display:
      ```bash
      conda install habitat-sim -c conda-forge -c aihabitat
      ```
      
   - Note: Build parameters can be chained together. For instance, to install habitat-sim with physics on headless machines:
      ```
      conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
      ```

Conda packages for older versions can installed by explicitly specifying the version, e.g. `conda install habitat-sim=0.1.6 -c conda-forge -c aihabitat`.

## Datasets
This project uses Habitat-Matterport 3D Research Dataset [HM3D](https://aihabitat.org/datasets/hm3d/) for generating the dataset and training the model. You can follow the steps provided in official HM3D [Github](https://github.com/facebookresearch/habitat-matterport3d-dataset) repository for getting access to the dataset.

[How To use other common supported datasets with Habitat-Sim](DATASETS.md).


## Testing

**To Verify the Habitat-sim setup**

1. Download some 3D assets using our python data download utility:
   - Download (testing) 3D scenes
       ```bash
      python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/
        ```
      
      **Note:**
       - The supported data file format is `.glb` to improt the envieonment in the habitat-simulator.
       - These testing scenes do not provide semantic annotations.
       - If you would like to test the semantic sensors via `example.py`, use the data from the Matterport3D dataset (see [Datasets](DATASETS.md)).


1. **Interactive testing**: Use the interactive viewer included with Habitat-Sim in python:

   ```
   #NOTE: depending on your choice of installation, you may need to add '/path/to/habitat-sim' to your PYTHONPATH.
   #e.g. from 'habitat-sim/' directory run 'export PYTHONPATH=$(pwd)'
   python equirec_navigation.py --scene /path/to/data/scene_datasets/habitat-test-scenes/file_name.glb
   ```

   To control an agent in the test scene:

   - `W/A/S/D` keys to move forward/left/backward/right
   -  arrow keys or mouse (LEFT click) to control gaze direction (look up/down/left/right)
   - `O` key to start recording the scene environment in equirectangular format.
   - `P` key to save the recording in the current directory which can be used to generate a sparse 3D point cloud map for the Robot/agent to train and navigate.

 Additionally, `--save_png` can be used to output agent visual observation frames of the physical scene to the current directory.


## Results:

The below results displays the output equirectangular video recorded from the sensor mounted on the agent traversing in the scene whcih can be used to generate the point cloud map of the environment by incoorporating SLAM pipeline.

<p align="center">
  <img src="output_data/equirec_output.gif" alt="Undistorted" width="700"/>
</p>


## Documentation

Browse the online [Habitat-Sim documentation](https://aihabitat.org/docs/habitat-sim/index.html).

To find the answers try asking the developers and community on our [Discussions forum](https://github.com/facebookresearch/habitat-lab/discussions).


## Citing Habitat

```
@inproceedings{szot2021habitat,
  title     =     {Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
  author    =     {Andrew Szot and Alex Clegg and Eric Undersander and Erik Wijmans and Yili Zhao and John Turner and Noah Maestre and Mustafa Mukadam and Devendra Chaplot and Oleksandr Maksymets and Aaron Gokaslan and Vladimir Vondrus and Sameer Dharur and Franziska Meier and Wojciech Galuba and Angel Chang and Zsolt Kira and Vladlen Koltun and Jitendra Malik and Manolis Savva and Dhruv Batra},
  booktitle =     {Advances in Neural Information Processing Systems (NeurIPS)},
  year      =     {2021}
}

@inproceedings{habitat19iccv,
  title     =     {Habitat: {A} {P}latform for {E}mbodied {AI} {R}esearch},
  author    =     {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and Yili Zhao and Erik Wijmans and Bhavana Jain and Julian Straub and Jia Liu and Vladlen Koltun and Jitendra Malik and Devi Parikh and Dhruv Batra},
  booktitle =     {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =     {2019}
}
```

