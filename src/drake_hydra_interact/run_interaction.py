#! /usr/bin/env python3

import sys
import os
import numpy as np
import random
import time
from threading import Lock
from termcolor import colored
import yaml

# ROS
import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import tf2_ros
import tf
import razer_hydra.msg

import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf

import pydrake
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    DiagramBuilder,
    FindResourceOrThrow,
    Quaternion,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SceneGraph,
    Simulator,
    SpatialInertia,
    Sphere,
    UniformGravityFieldElement,
    UnitInertia
)

from pydrake.all import (
    InverseKinematics,
    SnoptSolver
)

from drake_hydra_interact.hydra_system import HydraInteractionLeafSystem

help_string = '''
    The current wand pose will appear in the meshcat visualizer as a ball.
    The digital trigger grabs objects, softly welding (position + orientation)
    to the gripper for the duration of the trigger press.
'''


def ros_tf_to_rigid_transform(msg):
    return RigidTransform(
        p=[msg.translation.x, msg.translation.y, msg.translation.z],
        R=RotationMatrix(Quaternion(msg.rotation.w, msg.rotation.x,
                                    msg.rotation.y, msg.rotation.z)))


def save_config(all_object_instances, qf, filename):
    output_dict = {"n_objects": len(all_object_instances)}
    for k in range(len(all_object_instances)):
        offset = k*7
        pose = qf[(offset):(offset+7)]
        output_dict["obj_%04d" % k] = {
            "class": all_object_instances[k][0],
            "pose": pose.tolist(),
            "params": [],
            "params_names": []
        }
    with open(filename, "a") as file:
        yaml.dump({"env_%d" % int(round(time.time() * 1000)):
                   output_dict},
                   file)

def setup_environment():
    pass

def do_main():
    rospy.init_node('run_interaction', anonymous=False)
    
    for outer_iter in range(200):
        try:
            builder = DiagramBuilder()
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.0025))

            # Add ground
            world_body = mbp.world_body()
            ground_shape = Box(2., 2., 2.)
            ground_body = mbp.AddRigidBody("ground", SpatialInertia(
                mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
            mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                           RigidTransform(p=[0, 0, -1]))
            mbp.RegisterVisualGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
                np.array([0.5, 0.5, 0.5, 1.]))
            mbp.RegisterCollisionGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
                CoulombFriction(0.9, 0.8))

            parser = Parser(mbp, scene_graph)

            dish_bin_model = "/home/gizatt/projects/scene_generation/models/dish_models/bus_tub_01_decomp/bus_tub_01_decomp.urdf"
            cupboard_model = "/home/gizatt/projects/scene_generation/models/dish_models/shelf_two_levels.sdf"
            candidate_model_files = {
                #"mug": "/home/gizatt/drake/manipulation/models/mug/mug.urdf",
                "mug_1": "/home/gizatt/projects/scene_generation/models/dish_models/mug_1_decomp/mug_1_decomp.urdf",
                #"plate_11in": "/home/gizatt/drake/manipulation/models/dish_models/plate_11in_decomp/plate_11in_decomp.urdf",
                #"/home/gizatt/drake/manipulation/models/mug_big/mug_big.urdf",
                #"/home/gizatt/drake/manipulation/models/dish_models/bowl_6p25in_decomp/bowl_6p25in_decomp.urdf",
                #"/home/gizatt/drake/manipulation/models/dish_models/plate_8p5in_decomp/plate_8p5in_decomp.urdf",
            }

            # Decide how many of each object to add
            max_num_objs = 6
            num_objs = [np.random.randint(0, max_num_objs) for k in range(len(candidate_model_files.keys()))]

            # Actually produce their initial poses + add them to the sim
            poses = []  # [quat, pos]
            all_object_instances = []
            all_manipulable_body_ids = []
            total_num_objs = sum(num_objs)
            object_ordering = list(range(total_num_objs))
            k = 0
            random.shuffle(object_ordering)
            print("ordering: ", object_ordering)
            for class_k, class_entry in enumerate(candidate_model_files.items()):
                for model_instance_k in range(num_objs[class_k]):
                    class_name, class_path = class_entry
                    model_name = "%s_%d" % (class_name, model_instance_k)
                    all_object_instances.append([class_name, model_name])
                    model_id = parser.AddModelFromFile(class_path, model_name=model_name)
                    all_manipulable_body_ids += mbp.GetBodyIndices(model_id)

                    # Put them in a randomly ordered line, for placing
                    y_offset = (object_ordering[k] / float(total_num_objs) - 0.5)   #  RAnge -0.5 to 0.5
                    poses.append([
                        RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                        [-0.25, y_offset, 0.1]])
                    k += 1
                    #$poses.append([
                    #    RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                    #    [np.random.uniform(-0.2, 0.2), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.3)]])

            # Build a desk
            parser.AddModelFromFile(cupboard_model)
            mbp.WeldFrames(world_body.body_frame(), mbp.GetBodyByName("shelf_origin_body").body_frame(),
                           RigidTransform(p=[0.0, 0, 0.0]))
            #parser.AddModelFromFile(dish_bin_model)
            #mbp.WeldFrames(world_body.body_frame(), mbp.GetBodyByName("bus_tub_01_decomp_body_link").body_frame(),
            #               RigidTransform(p=[0.0, 0., 0.], rpy=RollPitchYaw(np.pi/2., 0., 0.)))

            mbp.AddForceElement(UniformGravityFieldElement())
            mbp.Finalize()

            hydra_sg_spy = builder.AddSystem(HydraInteractionLeafSystem(mbp, scene_graph, all_manipulable_body_ids=all_manipulable_body_ids))
            #builder.Connect(scene_graph.get_query_output_port(),
            #                hydra_sg_spy.get_input_port(0))
            builder.Connect(scene_graph.get_pose_bundle_output_port(),
                            hydra_sg_spy.get_input_port(0))
            builder.Connect(mbp.get_state_output_port(),
                            hydra_sg_spy.get_input_port(1))
            builder.Connect(hydra_sg_spy.get_output_port(0),
                            mbp.get_applied_spatial_force_input_port())

            visualizer = builder.AddSystem(MeshcatVisualizer(
                scene_graph,
                zmq_url="tcp://127.0.0.1:6000",
                draw_period=0.01))
            builder.Connect(scene_graph.get_pose_bundle_output_port(),
                            visualizer.get_input_port(0))

            diagram = builder.Build()

            diagram_context = diagram.CreateDefaultContext()
            mbp_context = diagram.GetMutableSubsystemContext(
                mbp, diagram_context)
            sg_context = diagram.GetMutableSubsystemContext(
                scene_graph, diagram_context)

            q0 = mbp.GetPositions(mbp_context).copy()
            for k in range(len(poses)):
                offset = k*7
                q0[(offset):(offset+4)] = poses[k][0]
                q0[(offset+4):(offset+7)] = poses[k][1]
            mbp.SetPositions(mbp_context, q0)
            simulator = Simulator(diagram, diagram_context)
            simulator.set_target_realtime_rate(1.0)
            simulator.set_publish_every_time_step(False)
            simulator.Initialize()

            ik = InverseKinematics(mbp, mbp_context)
            q_dec = ik.q()
            prog = ik.prog()

            def squaredNorm(x):
                return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2])
            for k in range(len(poses)):
                # Quaternion norm
                prog.AddConstraint(
                    squaredNorm, [1], [1], q_dec[(k*7):(k*7+4)])
                # Trivial quaternion bounds
                prog.AddBoundingBoxConstraint(
                    -np.ones(4), np.ones(4), q_dec[(k*7):(k*7+4)])
                # Conservative bounds on on XYZ
                prog.AddBoundingBoxConstraint(
                    np.array([-2., -2., -2.]), np.array([2., 2., 2.]),
                    q_dec[(k*7+4):(k*7+7)])

            def vis_callback(x):
                vis_diagram_context = diagram.CreateDefaultContext()
                mbp.SetPositions(diagram.GetMutableSubsystemContext(mbp, vis_diagram_context), x)
                pose_bundle = scene_graph.get_pose_bundle_output_port().Eval(diagram.GetMutableSubsystemContext(scene_graph, vis_diagram_context))
                context = visualizer.CreateDefaultContext()
                context.FixInputPort(0, AbstractValue.Make(pose_bundle))
                visualizer.Publish(context)

            prog.AddVisualizationCallback(vis_callback, q_dec)
            prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

            ik.AddMinimumDistanceConstraint(0.001, threshold_distance=1.0)

            prog.SetInitialGuess(q_dec, q0)
            print("Solving")
        #            print "Initial guess: ", q0
            start_time = time.time()
            solver = SnoptSolver()
            sid = solver.solver_type()
            # SNOPT
            prog.SetSolverOption(sid, "Print file", "test.snopt")
            prog.SetSolverOption(sid, "Major feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Major optimality tolerance", 1e-2)
            prog.SetSolverOption(sid, "Minor feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Scale option", 0)
            #prog.SetSolverOption(sid, "Elastic weight", 1e1)
            #prog.SetSolverOption(sid, "Elastic mode", "Yes")

            print("Solver opts: ", prog.GetSolverOptions(solver.solver_type()))
            result = solver.Solve(prog)
            print("Solve info: ", result)
            print("Solved in %f seconds" % (time.time() - start_time))
            print(result.get_solver_id().name())
            q0_proj = result.GetSolution(q_dec)
        #            print "Final: ", q0_proj
            mbp.SetPositions(mbp_context, q0_proj)

            simulator.StepTo(1000)
            raise StopIteration()

        except StopIteration:
            print(colored("Stopped, saving and restarting", 'yellow'))
            qf = mbp.GetPositions(mbp_context)

            # Decide whether to accept: all objs within bounds
            save = True
            for k in range(len(all_object_instances)):
                offset = k*7
                q = qf[offset:(offset + 7)]
                if q[4] <= -0.25 or q[4] >= 0.25 or q[5] <= -0.2 or q[5] >= 0.2 or q[6] <= -0.1:
                    save = False
                    break
            if save:
                print(colored("Saving", "green"))
                save_config(all_object_instances, qf, "mug_rack_environments_human.yaml")
            else:
                print(colored("Not saving due to bounds violation: " + str(q), "yellow"))

        except Exception as e:
            print(colored("Suffered other exception " + str(e), "red"))
            sys.exit(-1)
        except:
            print(colored("Suffered totally unknown exception! Probably sim.", "red"))

if __name__ == "__main__":
    do_main()
