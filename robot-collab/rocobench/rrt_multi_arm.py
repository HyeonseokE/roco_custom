import logging
import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from transforms3d import euler, quaternions
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.utils.transformations import mat_to_quat, quat_to_euler, euler_to_quat 

from rocobench.rrt import direct_path, smooth_path, birrt, NearJointsUniformSampler, CenterWaypointsUniformSampler
from rocobench.envs import SimRobot 
from rocobench.envs.env_utils import Pose


class MultiArmRRT:
    """ Stores the info for a group of arms and plan all the combined joints together """
    def __init__(
        self,
        physics,
        robots: Dict[str, SimRobot] = {},
        robot_configs: Dict[str, Dict[str, Any]] = {},
        seed: int = 0,
        graspable_object_names: Optional[Union[Dict[str, str], List[str]]] = None,
        allowed_collision_pairs: List[Tuple[int, int]] = [],
        inhand_object_info: Optional[Dict[str, Tuple]] = None,
    ):
        # embodiment 정보가 올바르게 전달되었는지 check.
        self.robots = robots
        if len(robots) == 0:
            assert len(robot_configs) > 0, "No robot config is passed in"
            print(
                "Warning: No robot is passed in, will use robot_configs to create robots"
            )
        
            for robot_name, robot_config in robot_configs.items():
                self.robots[robot_name] = SimRobot(physics, **robot_config)
        self.physics = physics 
        self.np_random = np.random.RandomState(seed)
        
        self.all_joint_names = []
        self.all_joint_ranges = []
        self.all_joint_idxs_in_qpos = []
        self.all_collision_link_names = []
        self.inhand_object_info = dict()

        # 각 로봇의 joint, collision_link 정보 저장.
        for name, robot in self.robots.items():
            self.all_joint_names.extend(
                robot.ik_joint_names
            ) 
            self.all_joint_idxs_in_qpos.extend(
                robot.joint_idxs_in_qpos
            )
            self.all_joint_ranges.extend(
                robot.joint_ranges
            )
            self.all_collision_link_names.extend(
                robot.collision_link_names
            )
        
        # in-hand object에 대한 정보 저장.
        self.set_inhand_info(physics, inhand_object_info)

        # 모든 로봇의 joint range 및 (min,max)값 구함.
        self.joint_minmax = np.array([jrange for jrange in self.all_joint_ranges])
        self.joint_ranges = self.joint_minmax[:, 1] - self.joint_minmax[:, 0]

        # assign a list of allowed grasp objects to each robot
        graspable_name_dict = dict()
        for robot_name in self.robots.keys():
            if type(graspable_object_names) is dict:
                assert robot_name in graspable_object_names, f"robot_name: {robot_name} not in graspable_object_names"
                graspable_name_dict[robot_name] = graspable_object_names[robot_name]

            elif type(graspable_object_names) is list:
                graspable_name_dict[robot_name] = graspable_object_names

        self.allowed_collision_pairs = allowed_collision_pairs

        # Find all sim objects that are not graspable
        self.set_ungraspable(graspable_name_dict)
    
    def set_inhand_info(self, physics, inhand_object_info: Optional[Dict[str, Tuple]] = None):
        """ Set the inhand object info """

        self.inhand_object_info = dict()
        if inhand_object_info is not None:
            for name, robot in self.robots.items():
                self.inhand_object_info[name] = None
                
                obj_info = inhand_object_info.get(name, None)
                
                if obj_info is not None:
                    if 'rope' in obj_info[0] or 'CB' in obj_info[0]:
                        continue
                    assert len(obj_info) == 3, f"inhand obj info: {obj_info} should be a tuple of (obj_body_name, obj_site_name, obj_joint_name)"
                    body_name, site_name, joint_name = obj_info
                    try:
                        mjsite = physics.data.site(site_name)
                        qpos_slice = physics.named.data.qpos._convert_key(joint_name) 
                    except: 
                        print(f"Error: site_name: {site_name} joint_name {joint_name} not found in mujoco model")
                        breakpoint() 
                    self.inhand_object_info[name] = (body_name, site_name, joint_name, (qpos_slice.start, qpos_slice.stop))
        return 
 
    
    def set_ungraspable(
        self, 
        graspable_object_dict: Optional[Dict[str, List[str]]]
    ):
        """ Find all sim objects that are not graspable """
        
        # 현재 sim에 있는 모든 bodies들 정보를 모음.(objects, link, joint 등)
        all_bodies = []
        for i in range(self.physics.model.nbody):
            all_bodies.append(self.physics.model.body(i))

        # ungraspable object 모으기    
        ungraspable_ids = [0]  # world

        # all robot link bodies are ungraspable:
        for name, robot in self.robots.items():
            ungraspable_ids.extend(
                robot.collision_link_ids
            )

        # append all children of ungraspable body
        # (동일한 물체의 부품은 같은 rootid를 갖음. ex 같은 로봇의 부품들은 같은rootid 갖음.)
        ungraspable_ids += [
            body.id for body in all_bodies if body.rootid[0] in ungraspable_ids
        ]
 
        # 만약 graspable_object_dict이 비었다면, ungraspable_id가 아닌 것들을 추가
        if graspable_object_dict is None or len(graspable_object_dict) == 0:
            graspable = set(
                [body.id for body in all_bodies if body.id not in ungraspable_ids]
            )
            ungraspable = set(ungraspable_ids)
            self.graspable_body_ids = {name: graspable for name in self.robots.keys()}
            self.ungraspable_body_ids = {name: ungraspable for name in self.robots.keys()}
        
        else: 
            # in addition to robots, everything else would be ungraspable if not in this list of graspable objects
            self.graspable_body_ids = {}
            self.ungraspable_body_ids = {}

            # graspable object에 대한 id, ungraspable object에 대한 id를 모음.
            for robot_name, graspable_object_names in graspable_object_dict.items():
                graspable_ids = [
                    body.id for body in all_bodies if body.name in graspable_object_names
                ]
                graspable_ids += [
                    body.id for body in all_bodies if body.rootid[0] in graspable_ids
                ]
                # graspable ids
                self.graspable_body_ids[robot_name] = set(graspable_ids)
                robot_ungraspable = ungraspable_ids.copy()
                robot_ungraspable += [
                    body.id for body in all_bodies if body.rootid[0] not in graspable_ids
                ]
                # ungraspable ids
                self.ungraspable_body_ids[robot_name] = set(ungraspable_ids)
            # breakpoint()

    def forward_kinematics_all(
        self,
        q: np.ndarray,
        physics = None,
        return_ee_pose: bool = False,
    ) -> Optional[Dict[str, Pose]]:
        
        if physics is None:
            physics = self.physics.copy(share_model=True)
        physics = physics.copy(share_model=True)
        
        # transform inhand objects!
        obj_transforms = dict()
        for robot_name, obj_info in self.inhand_object_info.items():
            gripper_pose = self.robots[robot_name].get_ee_pose(physics) # 현재 EE_6D_pose return.
            if obj_info is not None:
                body_name, site_name, joint_name, (start, end) = obj_info
                obj_quat = mat_to_quat(
                    physics.data.site(site_name).xmat.reshape((3, 3))
                )
                obj_pos = physics.data.site(site_name).xpos
                
                # 현재 EE와 inhanded_object간의 상대 자세 
                rel_rot = quaternions.qmult( 
                    quaternions.qinverse(
                        gripper_pose.orientation
                        ),
                    obj_quat,
                    )
                # 현재 EE와 inhanded_object간의 상대 위치
                rel_pos = obj_pos - gripper_pose.position
                obj_transforms[robot_name] = (rel_pos, rel_rot)
            else:
                obj_transforms[robot_name] = None

        # physics의 data.qpos를 target_qpos로 대체(움직여야할 qpos)
        physics.data.qpos[self.all_joint_idxs_in_qpos] = q
        physics.forward() # forward dynamics 계산

        # 바뀐 target_qpos를 로봇마다 모음.
        ee_poses = {}
        for robot_name, robot in self.robots.items():
            ee_poses[robot_name] = robot.get_ee_pose(physics)

        # also transform inhand objects!
        for robot_name, obj_info in self.inhand_object_info.items():
            if obj_info is not None:
                body_name, site_name, joint_name, (start, end) = obj_info
                rel_pos, rel_rot = obj_transforms[robot_name] 
                new_ee_pos = ee_poses[robot_name].position # target_qpos로 바뀐 EE위치
                new_ee_quat = ee_poses[robot_name].orientation # target_qpos로 바뀐 EE 방향
                target_pos = new_ee_pos + rel_pos # 바뀐 inhand object 위치
                target_quat = quaternions.qmult(new_ee_quat, rel_rot) # 바뀐 inhand object방향.

                # target_inhand_object의 IK를 풀어서 나온 physics내 모든 qpos를 반환.
                result = self.solve_ik(
                    physics,
                    site_name,
                    target_pos,
                    target_quat,
                    joint_names=[joint_name], 
                    max_steps=300,
                    inplace=0,   
                    )
                if result is not None:
                    # 바뀐 inhand_object qpos에 대해서만 현재 physics.qpos에 업데이트
                    new_obj_qpos = result.qpos[start:end]
                    physics.data.qpos[start:end] = new_obj_qpos
                    physics.forward() #forward_dynamics

        if return_ee_pose:
            return ee_poses
        
        # physics.step(10) # to make sure the physics is stable
        
        # 제안된 qpos의 FK를 풀어서 바뀐 위치로 update된 physics를 return.
        return physics # a copy of the original physics object 
 
    
    def check_joint_range(
        self, 
        physics,
        joint_names,
        qpos_idxs,
        ik_result,
        allow_err=0.03,
    ) -> bool:
        # 한 로봇을 구성하는 joint의 joint_range를 return.
        _lower, _upper = physics.named.model.jnt_range[joint_names].T

        # IK로부터 계산된 각 로봇 조인트가 움직여야할 각도(qpos)를 return.
        qpos = ik_result.qpos[qpos_idxs] # physics의 모든 joint_qpos중 로봇의 joint_qpos만을 인덱싱
        assert len(qpos) == len(_lower) == len(_upper), f"Shape mismatch: qpos: {qpos}, _lower: {_lower}, _upper: {_upper}"

        # 움직여야할 각도(qpos)가 joint_range를 넘는 지 여부 체크
        for i, name in enumerate(joint_names):
            if qpos[i] < _lower[i] - allow_err or qpos[i] > _upper[i] + allow_err:
                # print(f"Joint {name} out of range: {_lower[i]} < {qpos[i]} < {_upper[i]}")
                
                # qpos가 joint가 움직일 수 있는 범위를 넘으면 False
                return False 
        # qpos가 joint가 움직일 수 있는 범위 안이면 True
        return True

    def solve_ik(
        self,
        physics,
        site_name,
        target_pos,
        target_quat,
        joint_names, 
        tol=1e-14,
        max_steps=300,
        max_resets=20,
        inplace=True, 
        max_range_steps=0,
        qpos_idxs=None,
        allow_grasp=True,
        check_grasp_ids=None,
        check_relative_pose=False
    ):
        '''
        target_EE_pose의 IK를 풀어서 Arm의 join_angle을 구함.
        '''
        physics_cp = physics.copy(share_model=True)
        
        # 만약 IK를 계산한 qpos가 다른 arm과 충돌한다면 수행하는 함수.
        def reset_fn(physics):
            model = physics.named.model 
            _lower, _upper = model.jnt_range[joint_names].T
            
            # 현재 qpos를 조금 변경시켜서 로봇팔이 다른 각도로 target_EE_pose에 접근할 수 있게 유도.
            curr_qpos = physics.named.data.qpos[joint_names]
            # deltas = (_upper - _lower) / 2
            # new_qpos = self.np_random.uniform(low=_lower, high=_upper)
            new_qpos = self.np_random.uniform(low=curr_qpos-0.5, high=curr_qpos + 0.5)
            new_qpos = np.clip(new_qpos, _lower, _upper)
            # 현재 qpos를 조금 수정.
            physics.named.data.qpos[joint_names] = new_qpos
            physics.forward()

        for i in range(max_resets):
            # print(f"Resetting IK {i}")
            if i > 0:
                reset_fn(physics_cp)

            # target_6d_pose에 대한 IK 연산 수행. 
            # physics_cp안에 있는 모든 model의 qpos를 return.
            result = qpos_from_site_pose(
                physics=physics_cp,
                site_name=site_name,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=joint_names,
                tol=tol,
                max_steps=max_steps,
                inplace=True,
            )

            need_reset = False
            if result.success:
                in_range = True 
                collided = False
                if qpos_idxs is not None:
                    # IK을 풀어서 얻은 로봇의 qpos가 joint_range를 벗어나는 지 체크.
                    in_range = self.check_joint_range(physics_cp, joint_names, qpos_idxs, result)
                    ik_qpos = result.qpos.copy()

                    # 해당 로봇의 min, max_joint_range로 qpos클리핑.
                    _low, _high = physics_cp.named.model.jnt_range[joint_names].T
                    ik_qpos[qpos_idxs] = np.clip(
                        ik_qpos[qpos_idxs], _low, _high
                    )
                    # 협동 테스크에 참여하는 모든 로봇들의 joint_qpos로 인덱싱함.
                    ik_qpos = ik_qpos[self.all_joint_idxs_in_qpos]
                    # print('checking collision on IK result: step {}'.format(i))

                    # 모든 로봇 사이의 충돌여부를 체크
                    collided = self.check_collision(
                        physics=physics_cp,
                        robot_qpos=ik_qpos,
                        check_grasp_ids=check_grasp_ids,
                        allow_grasp=allow_grasp,
                        check_relative_pose=check_relative_pose,
                        )
                # 제안된 arm_joint_qpos가 collision이 있으면, 위에서 reset_fn 수행.
                need_reset = (not in_range) or collided

            else:
                need_reset = True

            # 제안된 arm_joint_qpos가 collision이 없으면, break.
            if not need_reset:
                break
        # img = physics_cp.render(camera_id='teaser', height=400, width=400)
        # plt.imshow(img)
        # plt.show()

        # 충돌이 발생하지 않은 qpos를 return.
        return result if result.success else None

    def inverse_kinematics_all(
        self,
        physics,
        ee_poses: Dict[str, Pose],
        inplace=False, 
        allow_grasp=True, 
        check_grasp_ids=None,
        check_relative_pose=False,
    ) -> Dict[str, Union[None, np.ndarray]]:
        '''
        target_ee_pose = {'Alice': Pose(pos=(-0.74,0.60,0.25),rot=(-0.05,0.02,0.63)), 
                         'Bob': Pose(pos=(-0.50,0.42,0.22),rot=(0.05,0.03,-1.54))}
        
        target_ee_pose의 IK를 풀어서 qpos를 return.                 
        '''
        if physics is None:
            physics = self.physics
        physics = physics.copy(share_model=True)
        results = dict() 

        # ee_poses = {'Alice': Pose(pos=(-0.74,0.60,0.25),rot=(-0.05,0.02,0.63)), 
        #             'Bob': Pose(pos=(-0.50,0.42,0.22),rot=(0.05,0.03,-1.54))}
        for robot_name, target_ee in ee_poses.items():

            # 각 로봇의 target_ee_pose구함(6D pose)
            assert robot_name in self.robots, f"robot_name: {robot_name} not in self.robots"
            robot = self.robots[robot_name]
            pos = target_ee.position # (x,y,z)
            quat = target_ee.orientation # (r,p,y)
            if robot.use_ee_rest_quat:
                quat = quaternions.qmult(
                    quat, robot.ee_rest_quat
                ), # TODO 
            # print(robot.ee_site_name, pos, quat, robot.joint_names)

            qpos_idxs = robot.joint_idxs_in_qpos # 현재 로봇의 joint_angle의 indices
            
            # IK을 계산해서 target_ee_pos를 위한 physics내 모든 물체들의 qpos를 구함
            result = self.solve_ik(
                physics=physics,
                site_name=robot.ee_site_name,
                target_pos=pos,
                target_quat=quat,
                joint_names=robot.ik_joint_names,
                tol=1e-14,
                max_steps=300,
                inplace=inplace,  
                qpos_idxs=qpos_idxs,
                allow_grasp=allow_grasp, 
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
            )

            if result is not None:
                # physic내의 모든 물체의 qpos 중 joint_angle_pos만 남김.
                result_qpos = result.qpos[qpos_idxs].copy()
                _lower, _upper = physics.named.model.jnt_range[robot.ik_joint_names].T
                result_qpos = np.clip(result_qpos, _lower, _upper)
                results[robot_name] = (result_qpos, qpos_idxs) 
            else:
                results[robot_name] = None

        # target_ee_pose를 위한 arm_joint의 qpos를 return.
        return results      


    def ee_l2_distance(
        self, 
        q1: np.ndarray, 
        q2: np.ndarray, 
        orientation_factor: float = 0.2
    ) -> float: 
        '''
        1. qpos1과 qpos2의 FK를 계산해서 각 위치,방향을 계산함.
        2. 두 위치의 L2-distance를 계산 후 return.
        '''
        # FK를 계산하여 qpos1의 EE_pose를 return.
        pose1s = self.forward_kinematics_all(q1, return_ee_pose=True) # {robotA: Pose1, robotB: Pose1}
        # FK를 계산하여 qpos2의 EE_pose를 return.
        pose2s = self.forward_kinematics_all(q2, return_ee_pose=True) # {robotA: Pose2, robotB: Pose2}
        assert pose1s is not None and pose2s is not None
        dist = 0

        # compute pair-wise distance between each robot's Pose1 and Pose2
        for robot_name in pose1s.keys():
            pose1 = pose1s[robot_name]
            pose2 = pose2s[robot_name]
            dist += pose1.distance(pose2, orientation_factor=orientation_factor)
        
        # return L2-distance between q1 and q2
        return dist

    def extend_ee_l2(
        self, 
        q1: np.ndarray, 
        q2: np.ndarray, 
        resolution: float = 0.006
    ) -> List[np.ndarray]:
        '''
        1. qpos1과 qpos2의 L2-distansce를 계산하여 동일하면 그냥 return.
        2. 아니라면, qpos1 + (1과2의 상대거리) x weight return. 
        '''
        dist = self.ee_l2_distance(q1, q2) # q1과 q2의 L2-distance를 계산.
        if dist == 0: # q1과q2가 동일하면, 같은 위치이므로 path = None
            return []
        step = resolution / dist # 매우작은 수 ex) 0.003
        
        # q1 + 상대위치*weight를 return <- weight가 0~1사이 값으로 target_qpos에서 조금 조정된 qpos들을 return. 
        return [(q2 - q1) * np.clip(t, 0, 1) + q1 for t in np.arange(0, 1 + step, step)]

    def allow_collision_pairs(
        self,
        physics: Any,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, List]] = None,
    ) -> Set[FrozenSet[int]]:
        
        """ Get the allowed collision pairs """ 

        # EE의 link들은 충돌을 허용
        allowed = set()
        for robot_name, robot in self.robots.items():
            # add the robot's set to the allowed set:
            allowed.update( 
                robot.ee_link_pairs
            )
        # 충돌허용파츠들도 충돌을 허용
        for id_pair in self.allowed_collision_pairs:
            allowed.add(frozenset([id_pair[0], id_pair[1]]))

        if allow_grasp:
            # if the robot is in contact with some allowed objects, allow the collision  
            # graspable objects들도 충돌을 허용
            for robot_name, robot in self.robots.items():
                assert robot_name in self.graspable_body_ids, f"Robot {robot_name} not found in graspable_body_ids"
                graspable_ids = self.graspable_body_ids[robot_name]
                
                if check_grasp_ids is not None:
                    assert robot_name in check_grasp_ids, f"Robot {robot_name} not found in check_grasp_ids"
                    # only find the desired grasp_id to allow collision
                    graspable_ids = check_grasp_ids[robot_name]

                for ee_id in robot.ee_link_body_ids:
                    for _id in graspable_ids: 
                        allowed.add(
                            frozenset([ee_id, _id])
                            ) 

                # dangerous: allow collision between all arm links and the object
                for arm_id in robot.collision_link_ids:
                    for _id in graspable_ids:
                        # if _id == 40: # broom in sweeping task
                        allowed.add(
                            frozenset([arm_id, _id])
                        )

        # 충돌이 허용된 파츠들을 return
        return allowed 

    def get_collided_links(
        self,
        qpos: Optional[np.ndarray] = None,
        physics = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, List]] = None,
        verbose: bool = False,
        show: bool = False,
    ) -> List[str]:
        """ Get the collided links """ 

        if physics is None:
            physics = self.physics.copy(share_model=True)

        # 제안된 qpos로의 FK를 풀어서 바뀐 위치로 update된 physics를 return.    
        physics = self.forward_kinematics_all(physics=physics, q=qpos, return_ee_pose=False) 
        
        # 두 로봇의 collidable_link_id를 return.
        robot_collison_ids = [physics.model.body(link_name).id for link_name in self.all_collision_link_names]
        # NOTE: cant allow grasped object to collide with other objects in the env
        
        allowed_collisions = self.allow_collision_pairs(
            physics, allow_grasp=allow_grasp, check_grasp_ids=check_grasp_ids
            ) # robot-to-object 
        collided_id1 = physics.model.geom_bodyid[physics.data.contact.geom1].copy()
        collided_id2 = physics.model.geom_bodyid[physics.data.contact.geom2].copy() 
        
        # collision이 허용된 부분들은 collied_id에서 제외.
        if len(allowed_collisions) > 0:
            undesired_mask = np.ones_like(collided_id1).astype(bool)

            for idx in range(len(collided_id1)):
                body1 = collided_id1[idx]
                body2 = collided_id2[idx]
                if frozenset([body1, body2]) in allowed_collisions:
                    undesired_mask[idx] = False

            collided_id1 = collided_id1[undesired_mask]
            collided_id2 = collided_id2[undesired_mask]
        
        # TODO: if an object is being grasped, don't allow it to collide with other objects
        # if allow_grasp and check_grasp_ids is not None:
        #     
        #     for robot_name, grasp_id in check_grasp_ids.items():
        #         graspable_ids = check_grasp_ids[robot_name]
        #         robot_collison_ids.extend(graspable_ids)
        all_pairs = set(zip(collided_id1, collided_id2))
        bad_pairs = set()

        for pair in all_pairs:
            # if pair[0] in robot_collison_ids or pair[1] in robot_collison_ids:
            root1 = physics.model.body(pair[0]).rootid 
            root2 = physics.model.body(pair[1]).rootid
            bad_pairs.add(
                (physics.model.body(root1).name, physics.model.body(root2).name)
            )
            # if (pair[0] == 62 and pair[1] == 64) or (pair[0] == 64 and pair[1] == 62):
            #     breakpoint()

        all_ids = set(collided_id1).union(set(collided_id2)) # could contain both object-to-object, robot-to-robot, etc
       
        # undesired_ids = set(robot_collison_ids).intersection(all_ids) 
        undesired_ids = all_ids
        
        # dist = np.linalg.norm(
        #     physics.data.site('robotiq_ee').xpos  - physics.data.site('panda_ee').xpos
        #     )
        # if dist > 0.8 or dist < 0.6:
        #     bad_pairs.add((dist, dist))

        # if a link is on robot AND it's in contact with something Not in the allowed_collisions
        # if np.linalg.norm(physics.data.body('red_cube').xpos  - physics.data.body('dustpan').xpos) < 0.1:
        # if 54 in collided_id1 or 54 in collided_id2:
        if len(undesired_ids) > 0 and show:
            print(bad_pairs)
            img_arr = np.concatenate(
                [
                     physics.render(camera_id=i, height=400, width=400,) for i in range(3)
                ]
                , axis=1
            )
            plt.imshow(img_arr)
            plt.show()
            
            qpos_str = " ".join(physics.data.qpos.astype(str))
            print(f"<key name='rrt_check' qpos='{qpos_str}'/>")
            breakpoint()
        # 충돌이 일어나는 model.body들을 return.
        return bad_pairs
    
    def check_relative_pose(
        self, 
        qpos: Optional[np.ndarray] = None,
        physics = None,
    ):  
        
        # get ee poses from qpos?
        poses_dict = self.forward_kinematics_all(q=qpos, physics=physics, return_ee_pose=True) # {robotA: Pose1, robotB: Pose1}
        alice_quat = np.array([7.07106781e-01, 1.73613722e-16, 1.69292055e-16, 7.07106781e-01])
        bob_quat = np.array([7.07106781e-01, 1.73613722e-16, 1.69292055e-16, 7.07106781e-01])
        rot_align = np.allclose(alice_quat, poses_dict['Alice'].orientation) and \
            np.allclose(bob_quat, poses_dict['Bob'].orientation)

        dist = np.linalg.norm(poses_dict["Alice"].position - poses_dict["Bob"].position)
        dist_align = 0.1 <= dist <= 0.4
        # print("===== dist", dist, dist_align)
        # print("===== rot_align", rot_align,  poses_dict['Alice'].orientation, poses_dict['Bob'].orientation)
        
        return 1 and dist_align
             

    def check_collision(
        self,
        robot_qpos: Optional[np.ndarray] = None,
        physics = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        verbose: bool = False,
        check_relative_pose: bool = False,
        show: bool = False,
    ) -> bool: 
 
        if check_relative_pose:
            passed = self.check_relative_pose(qpos=robot_qpos, physics=physics)
            if not passed:
                return True

        # 협동 task에 참여하는 로봇 팔 사이의 collision check.
        collided_links = self.get_collided_links(
            qpos=robot_qpos, 
            physics=physics,
            allow_grasp=allow_grasp,           
            check_grasp_ids=check_grasp_ids,
            verbose=verbose,
            show=show,
        ) 
        # if len(collided_links) > 0: 
        #     print("collided_link_ids", collided_links)
        #     for link_id in collided_links:
        #         link_name = self.physics.model.body(link_id).name
        #         print("collided_link_name", link_name)
        #     return True
        # print("collided_link_ids", collided_links)
        # for i in collided_links:
        #     print(self.physics.model.body(i).name)
        # if len(collided_links) > 0: 
        #     physics.named.data.qpos[self.all_joint_names] = robot_qpos
        #     physics.forward()
        #     img = physics.render(camera_id='teaser', height=400, width=400,)
        #     plt.imshow(img)
        #     plt.show()
        #     breakpoint()

        # 허용되지 않은 collision이 있는 경우, bad_pairs의 갯수 return
        bad = len(collided_links) > 0
        # collision발생 = 숫자(True), collision발생안함 = 0(False)
        return bad 


    
    def plan(
        self, 
        start_qpos: np.ndarray,  # can be either full length or just the desired qpos for the joints 
        goal_qpos: np.ndarray,
        init_samples: Optional[List[np.ndarray]] = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
    ) -> Tuple[Optional[List[np.ndarray]], str]:

        if len(start_qpos) != len(goal_qpos):
            return None, "RRT failed: start and goal configs have different lengths."
        if len(start_qpos) != len(self.all_joint_idxs_in_qpos):
            start_qpos = start_qpos[self.all_joint_idxs_in_qpos]
        if len(goal_qpos) != len(self.all_joint_idxs_in_qpos):
            goal_qpos = goal_qpos[self.all_joint_idxs_in_qpos]
  
        def collision_fn(q: np.ndarray, show: bool = False):
            return self.check_collision(
                robot_qpos=q,
                physics=self.physics,
                allow_grasp=allow_grasp,           
                check_grasp_ids=check_grasp_ids,  
                check_relative_pose=check_relative_pose,
                show=show,
                # detect_grasp=False, TODO?
            )

        if not skip_endpoint_collision_check:
            if collision_fn(start_qpos, show=1):
                # print("RRT failed: start qpos in collision.")
                return None, f"ReasonCollisionAtStart_time0_iter0"
            elif collision_fn(goal_qpos, show=1): 
                # print("RRT failed: goal qpos in collision.")
                return None, "ReasonCollisionAtGoal_time0_iter0"
        
        # 먼저, 1. 직관적 길찾기알고리즘으로 path를 찾고, 만약 실패(collision)시, 2. 랜덤샘플링기반 RRT로 길찾기 
        paths, info = birrt(
                start_conf=start_qpos,
                goal_conf=goal_qpos,
                distance_fn=self.ee_l2_distance,
                sample_fn=CenterWaypointsUniformSampler(
                    bias=0.05,
                    start_conf=start_qpos,
                    goal_conf=goal_qpos,
                    numpy_random=self.np_random,
                    min_values=self.joint_minmax[:, 0],
                    max_values=self.joint_minmax[:, 1],
                    init_samples=init_samples,
                ),
                extend_fn=self.extend_ee_l2,
                collision_fn=collision_fn,
                iterations=800,
                smooth_iterations=200,
                timeout=timeout,
                greedy=True,
                np_random=self.np_random,
                smooth_extend_fn=self.extend_ee_l2,
                skip_direct_path=skip_direct_path,
                skip_smooth_path=skip_smooth_path, # enable to make sure it passes through the valid init_samples 
            )

        if paths is None:
            return None, f"RRT failed: {info}"
        
        # start_qpos부터 target_qpos까지 가는 path를 return.
        return paths, f"RRT succeeded: {info}"
 
    def plan_splitted(
        self, 
        start_qpos: np.ndarray,  # can be either full length or just the desired qpos for the joints 
        goal_qpos: np.ndarray,
        init_samples: Optional[List[np.ndarray]] = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        '''
        start_qpos부터 goal_qpos까지를 잘게 나눠서 path로 만듬.
        '''
        
        all_paths, all_info = [], []
        duration = 0 
        iteration = 0
        def collision_fn(q: np.ndarray, show: bool = False):
            '''
            collision이 발생하면 True, 아니면 False를 return.
            '''
            return self.check_collision(
                robot_qpos=q,
                physics=self.physics,
                allow_grasp=allow_grasp,           
                check_grasp_ids=check_grasp_ids,  
                check_relative_pose=check_relative_pose,
                show=show,
                # detect_grasp=False, TODO?
            )
        
        # still try direct path first 
        if not skip_direct_path:
            start_time = time()
            
            # start_qpos부터 goal_qpos까지의 세분화된 path를 얻음.
            path = direct_path(start_qpos, goal_qpos, self.extend_ee_l2, collision_fn)
            if path is not None:
                return path, f"ReasonDirect_time{time() - start_time}_iter1"

        
        if not skip_endpoint_collision_check:
            if collision_fn(goal_qpos, show=1): 
                print("RRT failed: goal qpos in collision.")
                return None, "ReasonCollisionAtGoal_time0_iter0"
            
            valid_init_samples = []
            for i, interm_goal_qpos in enumerate(init_samples):
                if not collision_fn(interm_goal_qpos, show=0): 
                    valid_init_samples.append(interm_goal_qpos)
                # return None, "RRT failed: goal qpos in collision."
                # omit this waypoint and try planning with pruned init_sample 
            print(f"Given waypoints: {len(init_samples)}, valid: {len(valid_init_samples)} points")
            init_samples = valid_init_samples
        
        for i, interm_goal_qpos in enumerate(init_samples[::-1] + [goal_qpos]):
            interm_start_qpos = start_qpos if i == 0 else init_samples[::-1][i-1]
            print("planning interm_start_qpos", i)
            if len(interm_start_qpos) != len(interm_goal_qpos):
                return None, "RRT failed: start and goal configs have different lengths."
            if len(interm_start_qpos) != len(self.all_joint_idxs_in_qpos):
                interm_start_qpos = interm_start_qpos[self.all_joint_idxs_in_qpos]
            if len(interm_goal_qpos) != len(self.all_joint_idxs_in_qpos):
                interm_goal_qpos = interm_goal_qpos[self.all_joint_idxs_in_qpos]
        
        
            if not skip_endpoint_collision_check:
                if collision_fn(interm_start_qpos):
                    return None, f"ReasonCollisionAtStart_time0_iter0"
                elif collision_fn(interm_goal_qpos): 
                    return None, f"ReasonCollisionAtGoal_time0_iter0"
            
            paths, info = birrt(
                    start_conf=interm_start_qpos,
                    goal_conf=interm_goal_qpos,
                    distance_fn=self.ee_l2_distance,
                    sample_fn=CenterWaypointsUniformSampler(
                        bias=0.05,
                        start_conf=interm_start_qpos,
                        goal_conf=interm_goal_qpos,
                        numpy_random=self.np_random,
                        min_values=self.joint_minmax[:, 0],
                        max_values=self.joint_minmax[:, 1],
                        init_samples=[],
                    ),
                    extend_fn=self.extend_ee_l2,
                    collision_fn=collision_fn,
                    iterations=800,
                    smooth_iterations=200,
                    timeout=timeout,
                    greedy=True,
                    np_random=self.np_random,
                    smooth_extend_fn=self.extend_ee_l2,
                    skip_direct_path=skip_direct_path,
                    skip_smooth_path=skip_smooth_path, # enable to make sure it passes through the valid init_samples 
                )
             
            sub_duration = float(info.split("time")[1].split("_")[0])
            sub_iteration = int(info.split("iter")[1].split("_")[0])
            reason = info.split("Reason")[1].split("_")[0]
                
            if paths is None: 
                return None, f"Reason{reason}_time{sub_duration}_iter{sub_iteration}" 
            all_paths.extend(paths)
            all_info.append(info)
            duration += sub_duration
            iteration += sub_iteration
        
        if skip_smooth_path:
            return all_paths, f"ReasonSuccess_time{duration}_iter{iteration}"
        
        
        print('begin smoothing')
        smoothed_paths = smooth_path(
            path=all_paths,
            extend_fn=self.extend_ee_l2,
            collision_fn=collision_fn,
            np_random=self.np_random,
            iterations=50,
        )
        print('done smoothing')
        
        return smoothed_paths, f"ReasonSmoothed_time{duration}_iter{iteration}"
 