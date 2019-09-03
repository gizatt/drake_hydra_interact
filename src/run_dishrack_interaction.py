#! /usr/bin/env python
from __future__ import print_function

import sys
import os
import numpy as np
import random
import time

# ROS
#import rospy
#import sensor_msgs.msg
#import geometry_msgs.msg
#import tf2_ros
#import tf
#import razer_hydra.msg

import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf

import pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere,
    QueryObject
)
from pydrake.math import (RollPitchYaw, RotationMatrix, RigidTransform)
from pydrake.multibody.math import (SpatialForce)
from pydrake.multibody.tree import (
    BodyIndex,
    ForceElement,
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    ExternallyAppliedSpatialForce,
    VectorExternallyAppliedSpatialForced,
    MultibodyPlant,
)

from pydrake.forwarddiff import gradient
from pydrake.multibody.parsing import Parser
from pydrake.multibody.inverse_kinematics import InverseKinematics
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.ipopt import (IpoptSolver)
from pydrake.solvers.nlopt import (NloptSolver)
from pydrake.solvers.snopt import (SnoptSolver)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, BasicVector, DiagramBuilder, LeafSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle


help_string = '''
    The current wand pose will appear in the meshcat visualizer as a ball.
    The digital trigger grabs objects, softly welding (position + orientation)
    to the gripper for the duration of the trigger press.

    TODO:
    The analog stick (xy) + 2/4 (z) buttons moves the origin of the workspace
    region.
    3/1 buttons scale the workspace region.
'''


def ros_tf_to_rigid_transform(msg):
    return RigidTransform(
        p=[msg.translation.x, msg.translation.y, msg.translation.z],
        R=RotationMatrix(Quaternion(msg.rotation.w, msg.rotation.x,
                                    msg.rotation.y, msg.rotation.z)))

class HydraInteractionLeafSystem(LeafSystem):
    ''' Handles comms with the Hydra, and uses QueryObject inputs from the SceneGraph
    to pick closests points when required. Passes this information to the HydraInteractionForceElement
    at every update tick.'''
    def __init__(self, mbp, sg, zmq_url="default"):
        LeafSystem.__init__(self)

        self.set_name('HydraInteractionLeafSystem')

        # Pose bundle (from SceneGraph) input port.
        self.DeclareAbstractInputPort("query_object",
                                      AbstractValue.Make(sg.get_query_output_port().Eval(sg.CreateDefaultContext())))
        self.DeclareAbstractInputPort("pose_bundle",
                                      AbstractValue.Make(PoseBundle(0)))
        self.DeclareVectorInputPort("robot_state", BasicVector(mbp.num_positions() + mbp.num_velocities()))
        self.DeclareAbstractOutputPort(
                        "spatial_forces_vector",
                        lambda: AbstractValue.Make(VectorExternallyAppliedSpatialForced()),
                        self.DoCalcAbstractOutput)
        self.DeclarePeriodicPublish(0.01, 0.0)


        if zmq_url == "default":
            zmq_url = "tcp://127.0.0.1:6000"
        if zmq_url is not None:
            print("Connecting to meshcat-server at zmq_url=" + zmq_url + "...")
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        fwd_pt_in_hydra_frame = RigidTransform(p=[0.0, 0.0, 0.0])
        self.vis["hydra_origin"]["hand"].set_object(meshcat_geom.ObjMeshGeometry.from_file(os.path.join(os.getcwd(), "hand-regularfinal-scaled-1.obj")))

        self.vis["hydra_origin"]["hand"].set_transform(meshcat_tf.compose_matrix(scale=[0.001, 0.001, 0.001], angles=[np.pi/2, 0., np.pi/2], translate=[-0.25, 0., 0.]))
        #self.vis["hydra_origin"]["center"].set_object(meshcat_geom.Sphere(0.02))
        #self.vis["hydra_origin"]["center"].set_transform(meshcat_tf.translation_matrix([-0.025, 0., 0.]))
        #self.vis["hydra_origin"]["mid"].set_object(meshcat_geom.Sphere(0.015))
        #self.vis["hydra_origin"]["mid"].set_transform(meshcat_tf.translation_matrix([0.0, 0., 0.]))
        #self.vis["hydra_origin"]["fwd"].set_object(meshcat_geom.Sphere(0.01))
        #self.vis["hydra_origin"]["fwd"].set_transform(fwd_pt_in_hydra_frame.matrix())
        #self.vis["hydra_grab"].set_object(meshcat_geom.Sphere(0.01), 
        #                                  meshcat_geom.MeshLambertMaterial(
        #                                     color=0xff22dd,
        #                                     alphaMap=0.1))
        self.vis["hydra_grab"]["grab_point"].set_object(meshcat_geom.Sphere(0.01), 
                                                   meshcat_geom.MeshLambertMaterial(
                                                      color=0xff22dd,
                                                      alphaMap=0.1))
        # Hide it sketchily
        self.vis["hydra_grab"].set_transform(meshcat_tf.translation_matrix([0., 0., -1000.]))

        # State for selecting objects
        self.grab_needs_update = False
        self.grab_in_progress = False
        self.grab_update_hydra_pose = None
        self.selected_body = None
        self.selected_pose_in_body_frame = None
        self.desired_pose_in_world_frame = None
        self.stop = False

        # Set up subscription to Razer Hydra
        self.mbp = mbp
        self.mbp_context = mbp.CreateDefaultContext()
        self.sg = sg
        self.hasNewMessage = False
        self.lastMsg = None
        self.hydra_origin = RigidTransform(p=[1.0, 0., -0.1],
                                   rpy=RollPitchYaw([0., 0., 0.]))
        self.hydra_prescale = 3.0

        self.hydraSubscriber = rospy.Subscriber("/hydra_calib", razer_hydra.msg.Hydra, self.callback, queue_size=1)
        print("Waiting for hydra startup...")
        while not self.hasNewMessage  and not rospy.is_shutdown():
            rospy.sleep(0.01)
        print("Got hydra.")

    def DoCalcAbstractOutput(self, context, y_data):
        if self.selected_body and self.grab_in_progress:
            # Simple inverse dynamics PD rule to drive object to desired pose.
            body = self.selected_body

            # Load in robot state
            x_in = self.EvalVectorInput(context, 2).get_value()
            self.mbp.SetPositionsAndVelocities(self.mbp_context, x_in)
            TF_object = self.mbp.EvalBodyPoseInWorld(self.mbp_context, body)
            xyz = TF_object.translation()
            R = TF_object.rotation().matrix()
            TFd_object = self.mbp.EvalBodySpatialVelocityInWorld(self.mbp_context, body)
            xyzd = TFd_object.translational()
            Rd = TFd_object.rotational()

            desired_delta_pose = self.grab_update_hydra_pose.inverse().multiply(self.desired_pose_in_world_frame)

            # Apply this delta to the current object pose to get the desired final
            # pose.
            # Add the translations straight-up
            xyz_desired = self.desired_pose_in_world_frame.translation()

            # Regress xyz back to just the hydra pose in the attraction case
            if self.freeze_rotation:
                self.grab_update_hydra_pose = self.desired_pose_in_world_frame
                self.selected_body_init_offset = TF_object
                R_desired = self.selected_body_init_offset.rotation().matrix()
            else:
                R_desired = desired_delta_pose.matrix()[:3, :3].dot(self.selected_body_init_offset.rotation().matrix())
            # Could also pull the rotation back, but it's kind of nice to be able to recenter the object
            # without messing up a randomized rotation.
            #R_desired = (self.desired_pose_in_world_frame.rotation().matrix()*self.attract_factor +
            #             R_desired*(1.-self.attract_factor))

            # Apply PD in cartesian space
            xyz_e = xyz_desired - xyz
            xyzd_e = -xyzd
            f = 100.*xyz_e + 10.*xyzd_e

            R_err_in_body_frame = np.linalg.inv(R).dot(R_desired)
            aa = AngleAxis(R_err_in_body_frame)
            tau_p = R.dot(aa.axis()*aa.angle())
            tau_d = -Rd
            tau = tau_p + 0.05*tau_d

            exerted_force = SpatialForce(tau=tau, f=f)

            out = ExternallyAppliedSpatialForce()
            out.F_Bq_W = exerted_force
            out.body_index = self.selected_body.index()
            y_data.set_value(VectorExternallyAppliedSpatialForced([out]))
        else:
            y_data.set_value(VectorExternallyAppliedSpatialForced([]))


    def DoPublish(self, context, event):
        # TODO(russt): Copied from meshcat_visualizer.py. 
        # Change this to declare a periodic event with a
        # callback instead of overriding DoPublish, pending #9992.
        LeafSystem.DoPublish(self, context, event)

        if self.stop:
            self.stop = False
            if context.get_time() > 0.5:
                raise StopIteration

        query_object = self.EvalAbstractInput(context, 0).get_value()
        pose_bundle = self.EvalAbstractInput(context, 1).get_value()
        x_in = self.EvalVectorInput(context, 2).get_value()
        self.mbp.SetPositionsAndVelocities(self.mbp_context, x_in)
        
        if self.grab_needs_update:
            hydra_tf = self.grab_update_hydra_pose
            self.grab_needs_update = False
            # If grab point is colliding...
            #print [x.distance for x in query_object.ComputeSignedDistanceToPoint(hydra_tf.matrix()[:3, 3])]
            # Always just grab object 2. (The first object on the table.)
            self.selected_body = self.mbp.get_body(BodyIndex(2))
            self.selected_body_init_offset = self.mbp.EvalBodyPoseInWorld(
                self.mbp_context, self.selected_body)


    def callback(self, msg):
        ''' Control mapping: 
            Buttons: [Digital trigger, 1, 2, 3, 4, start, joy click]
            Digital trigger: buttons[0]
            Analog trigger: trigger
            Joy: +x is right, +y is fwd
        '''
        self.lastMsg = msg
        self.hasNewMessage = True

        pad_info = msg.paddles[0]
        hydra_tf_uncalib = ros_tf_to_rigid_transform(pad_info.transform)
        hydra_tf_uncalib.set_translation(hydra_tf_uncalib.translation()*self.hydra_prescale)
        hydra_tf = self.hydra_origin.multiply(hydra_tf_uncalib)
        self.desired_pose_in_world_frame = hydra_tf
        self.vis["hydra_origin"].set_transform(hydra_tf.matrix())


        # Interpret various buttons for changing to scaling
        if pad_info.buttons[0] and not self.grab_in_progress:
            print("Grabbing")
            self.grab_update_hydra_pose = hydra_tf
            self.grab_needs_update = True
            self.grab_in_progress = True
        elif self.grab_in_progress and not pad_info.buttons[0]:
            self.grab_in_progress = False

        self.freeze_rotation = pad_info.trigger > 0.15

        if pad_info.buttons[5]:
            self.stop = True

        if pad_info.buttons[1]:
            # Scale down
            self.hydra_prescale = max(0.01, self.hydra_prescale * 0.98)
            print("Updated scaling to ", self.hydra_prescale)
        if pad_info.buttons[3]:
            # Scale up
            self.hydra_prescale = min(10.0, self.hydra_prescale * 1.02)
            print("Updated scaling to ", self.hydra_prescale)
        if pad_info.buttons[2]:
            # Translate down
            translation = self.hydra_origin.translation().copy()
            translation[2] -= 0.01
            print("Updated translation to ", translation)
            self.hydra_origin.set_translation(translation)
        if pad_info.buttons[4]:
            # Translate up
            translation = self.hydra_origin.translation().copy()
            translation[2] += 0.01
            print("Updated translation to ", translation)
            self.hydra_origin.set_translation(translation)

        if abs(pad_info.joy[0]) > 0.01 or abs(pad_info.joy[1]) > 0.01:
            # Translate up
            translation = self.hydra_origin.translation().copy()
            translation[1] -= pad_info.joy[0]*0.01
            translation[0] += pad_info.joy[1]*0.01
            print("Updated translation to ", translation)
            self.hydra_origin.set_translation(translation)


def do_main():
    #rospy.init_node('run_dishrack_interaction', anonymous=False)
    
    #np.random.seed(42)
    
    while (1):
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
        candidate_model_files = {
            #"mug": "/home/gizatt/drake/manipulation/models/mug/mug.urdf",
            "mug": "/home/gizatt/projects/scene_generation/models/dish_models/mug_1_decomp/mug_1_decomp.urdf",
            "plate_11in": "/home/gizatt/drake/manipulation/models/dish_models/plate_11in_decomp/plate_11in_decomp.urdf",
            #"/home/gizatt/drake/manipulation/models/mug_big/mug_big.urdf",
            #"/home/gizatt/drake/manipulation/models/dish_models/bowl_6p25in_decomp/bowl_6p25in_decomp.urdf",
            #"/home/gizatt/drake/manipulation/models/dish_models/plate_8p5in_decomp/plate_8p5in_decomp.urdf",
        }

        # Decide how many of each object to add
        max_num_objs = 4
        num_objs = [np.random.randint(0, max_num_objs) for k in range(len(candidate_model_files.keys()))]

        # Actually produce their initial poses + add them to the sim
        poses = []  # [quat, pos]
        all_object_instances = []
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
                parser.AddModelFromFile(class_path, model_name=model_name)

                # Put them in a randomly ordered line, for placing
                #y_offset = (object_ordering[k] / float(total_num_objs) - 0.5)   #  RAnge -0.5 to 0.5
                #poses.append([
                #    RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                #    [-0.25, y_offset, 0.1]])
                #k += 1
                poses.append([
                    RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                    [np.random.uniform(-0.2, 0.2), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.3)]])

        # Build a desk
        #parser.AddModelFromFile("cupboard_without_doors.sdf")
        #mbp.WeldFrames(world_body.body_frame(), mbp.GetBodyByName("cupboard_body").body_frame(),
        #               RigidTransform(p=[0.25, 0, 0.3995 + 0.016/2])))
        parser.AddModelFromFile(dish_bin_model)
        mbp.WeldFrames(world_body.body_frame(), mbp.GetBodyByName("bus_tub_01_decomp_body_link").body_frame(),
                       RigidTransform(p=[0.0, 0., 0.], rpy=RollPitchYaw(np.pi/2., 0., 0.)))

        mbp.AddForceElement(UniformGravityFieldElement())
        mbp.Finalize()

        #hydra_sg_spy = builder.AddSystem(HydraInteractionLeafSystem(mbp, scene_graph))
        #builder.Connect(scene_graph.get_query_output_port(),
        #                hydra_sg_spy.get_input_port(0))
        #builder.Connect(scene_graph.get_pose_bundle_output_port(),
        #                hydra_sg_spy.get_input_port(1))
        #builder.Connect(mbp.get_state_output_port(),
        #                hydra_sg_spy.get_input_port(2))
        #builder.Connect(hydra_sg_spy.get_output_port(0),
        #                mbp.get_applied_spatial_force_input_port())

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
            mbp.SetPositions(mbp_context, x)
            pose_bundle = scene_graph.get_pose_bundle_output_port().Eval(sg_context)
            context = visualizer.CreateDefaultContext()
            context.FixInputPort(0, AbstractValue.Make(pose_bundle))
            #print(pose_bundle.get_pose(0))
            visualizer.Publish(context)
            #print("Here")

        prog.AddVisualizationCallback(vis_callback, q_dec)
        prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

        ik.AddMinimumDistanceConstraint(0.001, threshold_distance=1.0)

        prog.SetInitialGuess(q_dec, q0)
        print("Solving")
    #            print "Initial guess: ", q0
        start_time = time.time()
        solver = SnoptSolver()
        #solver = NloptSolver()
        sid = solver.solver_type()
        # SNOPT
        prog.SetSolverOption(sid, "Print file", "test.snopt")
        prog.SetSolverOption(sid, "Major feasibility tolerance", 1e-3)
        prog.SetSolverOption(sid, "Major optimality tolerance", 1e-2)
        prog.SetSolverOption(sid, "Minor feasibility tolerance", 1e-3)
        prog.SetSolverOption(sid, "Scale option", 0)
        #prog.SetSolverOption(sid, "Elastic weight", 1e1)
        #prog.SetSolverOption(sid, "Elastic mode", "Yes")
        # NLOPT
        #prog.SetSolverOption(sid, "initial_step", 0.1)
        #prog.SetSolverOption(sid, "xtol_rel", 1E-2)
        #prog.SetSolverOption(sid, "xtol_abs", 1E-2)

        #prog.SetSolverOption(sid, "Major step limit", 2)

        print("Solver opts: ", prog.GetSolverOptions(solver.solver_type()))
        result = mp.Solve(prog)
        print("Solve info: ", result)
        print("Solved in %f seconds" % (time.time() - start_time))
        #print(IpoptSolver().Solve(prog))
        print(result.get_solver_id().name())
        q0_proj = result.GetSolution(q_dec)
    #            print "Final: ", q0_proj
        mbp.SetPositions(mbp_context, q0_proj)

        try:
            simulator.StepTo(100000)
        except StopIteration:
            print("Stopped, restarting")
        except Exception as e:
            print("Suffered other exception ", e)
            sys.exit(-1)

if __name__ == "__main__":
    do_main()
