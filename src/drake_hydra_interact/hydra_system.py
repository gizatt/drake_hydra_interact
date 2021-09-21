#! /usr/bin/env python3

import numpy as np
from threading import Lock
import logging

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
    AngleAxis,
    BasicVector,
    ExternallyAppliedSpatialForce,
    LeafSystem,
    PoseBundle,
    QueryObject,
    RollPitchYaw,
    RotationMatrix,
    RigidTransform,
    SpatialForce
)


class HydraInteractionLeafSystem(LeafSystem):
    ''' Handles comms with the Hydra, and uses QueryObject inputs from the SceneGraph
    to pick closests points when required. Passes this information to the HydraInteractionForceElement
    at every update tick.

    Construct by supplying the MBPlant and SceneGraph under sim + a list of the body IDs
    that are manipulable. Given ZMQ information, visualizes hand pose with a hand model.

    TODO: Hand geometric may want to be handed to SceneGraph to visualize; would need to
    investigate piping for manually specifying poses of objects not in MBP. '''
    def __init__(self, mbp, sg, all_manipulable_body_ids=[], zmq_url="default"):
        LeafSystem.__init__(self)
        self.all_manipulable_body_ids = all_manipulable_body_ids
        self.set_name('HydraInteractionLeafSystem')

        # Pose bundle (from SceneGraph) input port.
        #default_sg_context = sg.CreateDefaultContext()
        #print("Default sg context: ", default_sg_context)
        #query_object = sg.get_query_output_port().Eval(default_sg_context)
        #print("Query object: ", query_object)
        #self.DeclareAbstractInputPort("query_object",
        #                              AbstractValue.Make(query_object))
        self.pose_bundle_input_port = self.DeclareAbstractInputPort(
            "pose_bundle", AbstractValue.Make(PoseBundle(0)))
        self.robot_state_input_port = self.DeclareVectorInputPort(
            "robot_state", BasicVector(mbp.num_positions() + mbp.num_velocities()))
        self.spatial_forces_output_port = self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.DoCalcAbstractOutput)
        self.DeclarePeriodicPublish(0.01, 0.0)


        if zmq_url == "default":
            zmq_url = "tcp://127.0.0.1:6000"
        if zmq_url is not None:
            logging.info("Connecting to meshcat-server at zmq_url=" + zmq_url + "...")
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        fwd_pt_in_hydra_frame = RigidTransform(p=[0.0, 0.0, 0.0])
        self.vis["hydra_origin"]["hand"].set_object(
            meshcat_geom.ObjMeshGeometry.from_file(
                os.path.join(os.getcwd(), "hand-regularfinal-scaled-1.obj"))
        )

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
        self.freeze_rotation = False
        self.previously_freezing_rotation = False

        # Set up subscription to Razer Hydra
        self.mbp = mbp
        self.mbp_context = mbp.CreateDefaultContext()
        self.sg = sg
        self.hasNewMessage = False
        self.lastMsg = None
        self.hydra_origin = RigidTransform(p=[1.0, 0., -0.1],
                                   rpy=RollPitchYaw([0., 0., 0.]))
        self.hydra_prescale = 3.0

        self.callback_lock = Lock()
        self.hydraSubscriber = rospy.Subscriber("/hydra_calib", razer_hydra.msg.Hydra, self.callback, queue_size=1)
        logging.info("Waiting for hydra startup...")
        while not self.hasNewMessage  and not rospy.is_shutdown():
            rospy.sleep(0.01)
        logging.info("Got hydra.")

    def DoCalcAbstractOutput(self, context, y_data):
        self.callback_lock.acquire()

        if self.selected_body and self.grab_in_progress:
            # Simple inverse dynamics PD rule to drive object to desired pose.
            body = self.selected_body

            # Load in robot state
            x_in = self.EvalVectorInput(context, 1).get_value()
            self.mbp.SetPositionsAndVelocities(self.mbp_context, x_in)
            TF_object = self.mbp.EvalBodyPoseInWorld(self.mbp_context, body)
            xyz = TF_object.translation()
            R = TF_object.rotation().matrix()
            TFd_object = self.mbp.EvalBodySpatialVelocityInWorld(self.mbp_context, body)
            xyzd = TFd_object.translational()
            Rd = TFd_object.rotational()

            # Match the object position directly to the hydra position.
            xyz_desired = self.desired_pose_in_world_frame.translation()

            # Regress xyz back to just the hydra pose in the attraction case
            if self.previously_freezing_rotation != self.freeze_rotation:
                self.selected_body_init_offset = TF_object
                self.grab_update_hydra_pose = RigidTransform(self.desired_pose_in_world_frame)
            self.previously_freezing_rotation = self.freeze_rotation
            if self.freeze_rotation:
                R_desired = self.selected_body_init_offset.rotation().matrix()
            else:
                # Figure out the relative rotation of the hydra from its initial posture
                to_init_hydra_tf = self.grab_update_hydra_pose.inverse()
                desired_delta_rotation = to_init_hydra_tf.multiply(self.desired_pose_in_world_frame).matrix()[:3, :3]
                # Transform the current object rotation into the init hydra frame, apply that relative tf, and
                # then transform back
                to_init_hydra_tf_rot = to_init_hydra_tf.matrix()[:3, :3]
                R_desired = to_init_hydra_tf_rot.T.dot(
                    desired_delta_rotation.dot(to_init_hydra_tf_rot.dot(
                        self.selected_body_init_offset.rotation().matrix())))

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
            tau = tau_p + 0.1*tau_d

            exerted_force = SpatialForce(tau=tau, f=f)

            out = ExternallyAppliedSpatialForce()
            out.F_Bq_W = exerted_force
            out.body_index = self.selected_body.index()
            y_data.set_value([out])
        else:
            y_data.set_value([])
        self.callback_lock.release()

    def DoPublish(self, context, event):
        # TODO(russt): Copied from meshcat_visualizer.py. 
        # Change this to declare a periodic event with a
        # callback instead of overriding DoPublish, pending #9992.
        LeafSystem.DoPublish(self, context, event)

        self.callback_lock.acquire()

        if self.stop:
            self.stop = False
            if context.get_time() > 0.5:
                self.callback_lock.release()
                raise StopIteration

        #query_object = self.EvalAbstractInput(context, 0).get_value()
        pose_bundle = self.EvalAbstractInput(context, 0).get_value()
        x_in = self.EvalVectorInput(context, 1).get_value()
        self.mbp.SetPositionsAndVelocities(self.mbp_context, x_in)
        
        if self.grab_needs_update:
            hydra_tf = self.grab_update_hydra_pose
            self.grab_needs_update = False
            # If grab point is colliding...
            #print [x.distance for x in query_object.ComputeSignedDistanceToPoint(hydra_tf.matrix()[:3, 3])]
            # Find closest body to current pose

            grab_center = hydra_tf.matrix()[:3, 3]
            closest_distance = np.inf
            closest_body = self.mbp.get_body(BodyIndex(2))
            for body_id in self.all_manipulable_body_ids:
                body = self.mbp.get_body(body_id)
                offset = self.mbp.EvalBodyPoseInWorld(self.mbp_context, body)
                dist = np.linalg.norm(grab_center - offset.translation())
                if dist < closest_distance:
                    closest_distance = dist
                    closest_body = body

            self.selected_body = closest_body
            self.selected_body_init_offset = self.mbp.EvalBodyPoseInWorld(
                self.mbp_context, self.selected_body)
        self.callback_lock.release()

    def callback(self, msg):
        ''' Control mapping: 
            Buttons: [Digital trigger, 1, 2, 3, 4, start, joy click]
            Digital trigger: buttons[0]
            Analog trigger: trigger
            Joy: +x is right, +y is fwd
        '''
        self.callback_lock.acquire()

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
            logging.info("Grabbing.")
            self.grab_update_hydra_pose = hydra_tf
            self.grab_needs_update = True
            self.grab_in_progress = True
        elif self.grab_in_progress and not pad_info.buttons[0]:
            self.grab_in_progress = False
            self.selected_body = None

        self.freeze_rotation = pad_info.trigger > 0.15

        if pad_info.buttons[5]:
            self.stop = True

        # Optional: use buttons to adjust hydra-reality scaling.
        # Disabling for easier onboarding of new users...
        #if pad_info.buttons[1]:
        #    # Scale down
        #    self.hydra_prescale = max(0.01, self.hydra_prescale * 0.98)
        #    print("Updated scaling to ", self.hydra_prescale)
        #if pad_info.buttons[3]:
        #    # Scale up
        #    self.hydra_prescale = min(10.0, self.hydra_prescale * 1.02)
        #    print("Updated scaling to ", self.hydra_prescale)
        #if pad_info.buttons[2]:
        #    # Translate down
        #    translation = self.hydra_origin.translation().copy()
        #    translation[2] -= 0.01
        #    print("Updated translation to ", translation)
        #    self.hydra_origin.set_translation(translation)
        #if pad_info.buttons[4]:
        #    # Translate up
        #    translation = self.hydra_origin.translation().copy()
        #    translation[2] += 0.01
        #    print("Updated translation to ", translation)
        #    self.hydra_origin.set_translation(translation)
        #if abs(pad_info.joy[0]) > 0.01 or abs(pad_info.joy[1]) > 0.01:
        #    # Translate up
        #    translation = self.hydra_origin.translation().copy()
        #    translation[1] -= pad_info.joy[0]*0.01
        #    translation[0] += pad_info.joy[1]*0.01
        #    print("Updated translation to ", translation)
        #self.hydra_origin.set_translation(translation)
        self.callback_lock.release()