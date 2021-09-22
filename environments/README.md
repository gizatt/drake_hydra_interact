# Environment descriptions

Each subfolder should contain:
- An `environment_description.yaml` file, detailed below.
- Any number of self-contained SDF, URDF, mesh, texture, etc resources.

It may also contain yaml files listing observed environments, which are detailed below.

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

occurance:
  #                0   1   2   3   4   5   6   7
  obj_count_dist: [0., 0.5, 0.25, 1., 1., 1.]

placeable_objects:
  - model_file: "dish.sdf"
    metadata:
      foo: bar
```

`world_description` contains a list of (`model_file`, `pose`) pairs that describes the rigid geometry of the world. The body specified by "root_body_name" will be welded to the world. An additional `has_ground` field toggles whether a plane at z=0 is inserted as ground.

`occurance` describes how many objects to initialize in a scene. Currently, it takes the form of a categorical distribution over object counts with specified weights.

`placeable_objects` contains a list of `model_file, occurance`s that describe objects that should be placed into the world. Each of these model files (SDF or URDF) should contain *exactly one body*. Metadata gets copied out to exported scene descriptions for downstream use.


## Observed environment yaml

Example:
```
env_1632269755585:
  objects:
  - metadata:
      class: plate
    model_file: plates_cups_and_bowls/plates/Room_Essentials_Salad_Plate_Turquoise/model_simplified.sdf
    pose:
      rpy:
      - -0.004306018339555129
      - 0.014380971273873657
      - -2.7679342144005084
      xyz:
      - 0.4245758672485237
      - -0.20270672436767606
      - 0.015342650791231184
  world_description:
    has_ground: true
    models:
    - model_file: bin.sdf
      pose:
        rpy:
        - 0.0
        - 0.0
        - 0.0
        xyz:
        - 0.5
        - 0.0
        - 0.0
      root_body_name: bin_base
```

List of envs keyed by unique names at top level. Each env contains a list of objects, and a world description.