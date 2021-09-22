#! /usr/bin/env python3

import argparse
from copy import deepcopy
import sys
import os
import numpy as np
import random
import time
import logging

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# ROS
import rospy

import meshcat

import pydrake
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Box,
    ConnectMeshcatVisualizer,
    CoulombFriction,
    DiagramBuilder,
    FindResourceOrThrow,
    Quaternion,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SceneGraph,
    Simulator,
    SpatialInertia,
    Sphere,
    UnitInertia
)

from pydrake.all import (
    InverseKinematics,
    SnoptSolver
)

from drake_hydra_interact.hydra_system import HydraInteractionLeafSystem

def save_config(objects, world_description, filename):
    # Given an object dict (with pose annotated with xyz and rpy entries),
    # saves out a scene yaml dict.
    env_dict = {
        "env_%d" % int(round(time.time() * 1000)):
            {
                "world_description": world_description,
                "objects": objects
            }
    }
    with open(filename, "a") as file:
        yaml.dump(env_dict, file)

def setup_args():
    parser = argparse.ArgumentParser(description='Do interactive placement of objects.')
    parser.add_argument('environment_folder',
                        help='Path to folder containing environment_description.yaml')
    parser.add_argument('--no_hydra', action='store_true',
                        help='Disable hydra input, which also disables ROS requirement.')
    parser.add_argument('--timestep', default="0.001", type=float,
                        help='Drake MBP simulation timestep.')
    parser.add_argument('--zmq_url', default=None,
                        help='ZMQ url of meshcat server to connect to. See drake/MeshcatVisualizer')
    args = parser.parse_args()
    return args

def load_env_description(args):
    # Returns yaml dictionary from the environment_description.yaml specified
    # by the args.
    env_yaml_path = os.path.join(args.environment_folder, "environment_description.yaml")
    assert os.path.exists(env_yaml_path), "Env description not found at %s" % env_yaml_path
    with open(env_yaml_path, "r") as f:
        return yaml.load(f, Loader=Loader)

def add_ground(mbp):
    world_body = mbp.world_body()
    ground_shape = Box(10., 10., 2.)
    ground_body = mbp.AddRigidBody("ground", SpatialInertia(
        mass=100.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                   RigidTransform(p=[0, 0, -1]))
    mbp.RegisterVisualGeometry(
        ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
        np.array([0.5, 0.5, 0.5, 1.]))
    mbp.RegisterCollisionGeometry(
        ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
        CoulombFriction(0.9, 0.8))

def add_model(mbp, parser, name, model_path, tf, fixed, root_body_name=None):
    # Adds model from model_path to the mbp. Returns a list of
    # body ids added to the model. root_body_name (or an automatically-determined
    # root body) will be welded to the world if requested.
    model_id = parser.AddModelFromFile(model_path, model_name=name)

    # Figure out root body if we aren't given one.
    body_inds = mbp.GetBodyIndices(model_id)
    if root_body_name is None:
        assert len(body_inds) == 1, \
            "Please supply root_body_name for model with path %s" % model_path
        root_body = mbp.get_body(body_inds[0])
    else:
        root_body = mbp.GetBodyByName(
            name=root_body_name,
            model_instance=model_id)

    if fixed:
        mbp.WeldFrames(
            mbp.world_body().body_frame(),
            root_body.body_frame(),
            tf
        )
    else:
        mbp.SetDefaultFreeBodyPose(root_body, tf)

    return body_inds

def collect_placement(args):
    ''' Sets up a new MBP based on the environment folder specified in the args.
    Connects a Razer hydra (unless suppresed), and runs an interactive placement sim. '''
    try:
        # Load in YAML
        yaml_info = load_env_description(args)
    
        # Build MBP
        builder = DiagramBuilder()
        mbp, scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=args.timestep))

        # Add ground if requested
        if yaml_info["world_description"]["has_ground"] is True:
            add_ground(mbp)

        # Get ready to parse lots of model files.
        parser = Parser(mbp, scene_graph)

        # First load in the world models.
        for k, model_info in enumerate(yaml_info["world_description"]["models"]):
            tf = RigidTransform(
                p=model_info["pose"]["xyz"],
                rpy=RollPitchYaw(model_info["pose"]["rpy"])
            )
            root_body_name = None
            if "root_body_name" in model_info.keys():
                root_body_name = model_info["root_body_name"]
            model_file = os.path.join(args.environment_folder, model_info["model_file"])
            add_model(
                mbp, parser, "world_model_%d" % k, model_file,
                tf=tf, fixed=True, root_body_name=root_body_name
            )

        # Decide how many objects to spawn.
        count_weights = np.array(yaml_info["occurance"]["obj_count_dist"])
        count_weights /= count_weights.sum()
        n_objects = np.random.choice(range(len(count_weights)), p=count_weights)
        rospy.loginfo("Sampling %d objects" % n_objects)

        # Pick a random object n_object times.
        objects = []
        n_available_objects = len(yaml_info["placeable_objects"])
        for k in range(n_objects):
            obj_k = np.random.randint(n_available_objects)
            objects.append(deepcopy(yaml_info["placeable_objects"][obj_k]))

        # Add objects to the sim in a -y/+y line.
        all_manipulable_body_ids = []

        for k, model_info in enumerate(objects):
            y_offset = ((k + 1) // 2) * 0.25  # [ 0, .2, .2, .4, .4, ...]
            y_offset *= (k % 2)*2. - 1. # [0, .2, -.2, .4, -.4, ...]
            tf = RigidTransform(
                p=np.array([0., y_offset, 0.2]),
                # Not true uniform random rotations, but it's not important here
                rpy=RollPitchYaw(np.random.uniform(0., 2*np.pi, size=3))
            )
            model_file = os.path.join(args.environment_folder, model_info["model_file"])
            root_body_name = None
            if "root_body_name" in model_info.keys():
                root_body_name = model_info["root_body_name"]
            new_body_inds = add_model(
                mbp, parser, "manip_model_%d" % k, model_file,
                tf=tf, fixed=False, root_body_name=root_body_name
            )
            all_manipulable_body_ids += new_body_inds
            assert len(new_body_inds) == 1, new_body_inds
            model_info["body_id"] = new_body_inds[0]

        mbp.Finalize()

        if not args.no_hydra:
            hydra_sg_spy = builder.AddSystem(HydraInteractionLeafSystem(mbp, scene_graph, all_manipulable_body_ids=all_manipulable_body_ids))
            builder.Connect(scene_graph.get_query_output_port(),
                            hydra_sg_spy.get_input_port(0))
            builder.Connect(mbp.get_state_output_port(),
                            hydra_sg_spy.get_input_port(1))
            builder.Connect(hydra_sg_spy.get_output_port(0),
                            mbp.get_applied_spatial_force_input_port())

        visualizer = ConnectMeshcatVisualizer(
            builder, scene_graph,
            zmq_url=args.zmq_url,
            draw_period=0.0333
        )

        diagram = builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        mbp_context = diagram.GetSubsystemContext(mbp, diagram_context)
        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(1.0)
        simulator.set_publish_every_time_step(False)
        simulator.Initialize()
        simulator.AdvanceTo(1000.)
        raise StopIteration()

    except StopIteration:
        rospy.loginfo("Stopped, saving and restarting")

        # Build pose entries for the objects
        for model_info in objects:
            # Remove body index once we're used it, since we don't
            # need it and can't save it.
            tf = mbp.GetFreeBodyPose(
                mbp_context, mbp.get_body(model_info.pop("body_id"))
            )
            model_info["pose"] = {
                "xyz": tf.translation().tolist(),
                "rpy": RollPitchYaw(tf.rotation()).vector().tolist()
            }
        save_config(
            objects, yaml_info["world_description"],
            os.path.join(args.environment_folder, "saved_scenes.yaml")
        )

    except Exception as e:
        rospy.logerr("Suffered other exception " + str(e))
        sys.exit(-1)
    except:
        rospy.logerr("Suffered totally unknown exception! Probably sim.")

def do_main_loop(args):
    rospy.init_node('run_interaction', anonymous=False)
    for k in range(10):
        collect_placement(args)

if __name__ == "__main__":
    args = setup_args()
    do_main_loop(args)
