# Environment descriptions

Each subfolder should contain:
- An `environment_description.yaml` file, detailed below.
- Any number of self-contained SDF, URDF, mesh, texture, etc resources.

## `environment_description.yaml`

Example:

```
world_description:
  has_ground: True
  - model_file: "bin.sdf"
    root_body_name: "root_body_name"
    pose:
      xyz: [0., 1., 2.]
      rpy: [0., 3.14, 0.]
  - model_file: ""

placeable_objects:
  - model_file: "dish.sdf"
    occurance:
      min: 0
      max: 3
      rate: 0.5
```

`world_description` contains a list of (`model_file`, `pose`) pairs that describes the rigid geometry of the world. The body specified by "root_body_name" will be welded to the world. An additional `has_ground` field toggles whether a plane at z=0 is inserted as ground.

`placeable_objects` contains a list of `model_file, occurance`s that describe objects that should be placed into the world. Each of these model files (SDF or URDF) should contain *exactly one body*. Occurance describes the min, max, and geometric rate of a bounded geometric distribution, which is used to decide how many of each object type to spawn into a given world.
