import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Dict, Set
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from pydantic import dataclasses, validator
import matplotlib.pyplot as plt

from rocobench.envs import SimAction, EnvState, SimRobot
from rocobench.envs.env_utils import Pose
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.subtask_plan import LLMPathPlan


class PlannedPathPolicy:
    """
    Takes in a series of LLM-proposed plans, i.e. a path of desired ee poses for each robot and where should it grasp/resealse objects
    Use these plans to compute the desired joint position waypoints via IK
    Use MultiArmRRT to plan and interpolate between the waypoints
    By default, each RRT planning step only cares about going from start to end without arms colliding,
    so the intermediate GPT-proposed waypoints may get skipped in the final motion plan.
    The plan for each robot may end with a target object to grasp.
    Note the assumption: each LLMPathPlan can grasp at most one object per robot, so a pick-and-place motion would need two LLMPathPlan's to complete. 
    """
    def __init__(
        self,
        physics,
        robots: Dict[str, SimRobot],
        path_plan: LLMPathPlan,  
        control_freq: int = 20,
        close_loop: bool = False,
        use_weld: bool = True,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        graspable_object_names: Optional[Union[Dict[str, str], List[str]]] = None,
        check_relative_pose: bool = False,
        allowed_collision_pairs: Optional[List[Tuple[int, int]]] = None,
        plan_splitted: bool = False,
        timeout: int = 200,
    ):
        self.robot_names = robots.keys()
        self.robots = robots
        physics = physics.copy(share_model=True)
        self.graspable_object_names = graspable_object_names

        # RRT_Planner init
        self.rrt_planner = MultiArmRRT(
            physics=physics,
            robots=robots,
            graspable_object_names=graspable_object_names,
            allowed_collision_pairs=allowed_collision_pairs,
            inhand_object_info=path_plan.get_inhand_obj_info(physics),
            )

        self.robots = robots
        self.rrt_plan_results = None
        self.path_plan = path_plan 
        self.control_freq = control_freq
        
        self.close_loop = close_loop # need to re-plan if close_loop is True
        self.check_relative_pose = check_relative_pose
        
        # check if target object and site are valid
        self.use_weld = use_weld # need to touch model.eq_active
        self.tograsp = self.parse_llm_plan_for_grasp(physics, path_plan)# parse each object to grasp/release
        self.inhand = path_plan.get_inhand_ids(physics).copy() # return inhanded objects
        self.grasp_allowed = path_plan.get_allowed_collision_ids(physics).copy() # includes both inhand and tograsp bodies
        self.allowed_collision_pairs = allowed_collision_pairs
        
        # natural_language plan을 qpos로 만듬
        self.parse_llm_plan_to_qpos(
            physics, path_plan, update=True
            )
        self.action_buffer = []
        self.action_idx = 0
        self.skip_direct_path = skip_direct_path # enforces the planner to go through the valid waypoints 
        self.skip_smooth_path = skip_smooth_path # skip smoothing the path, useful for debugging
        self.plan_splitted = plan_splitted # if True, the plan is splitted into two parts, one for each robot
        self.timeout = timeout # timeout for each planning step, in number of planning steps


    def ik_ee_poses_to_qpos(self, physics, ee_poses: Dict[str, Pose]) -> Dict[str, np.ndarray]:
        """
        Computes the joint positions for each robot to achieve the desired ee poses
        """
        for _name in self.robot_names:
            assert _name in ee_poses.keys(), f"missing robot name {_name} in ee_poses"

        # 환경내의 모든 물체의 current_qpos를 구함.
        full_qpos_result = physics.data.qpos.copy()

        # target_EE_pose의 IK를 풀어서 로봇arm의 target_joint_angles(qpos)을 구함.
        qpos_target_dict = self.rrt_planner.inverse_kinematics_all(
            physics=physics,
            ee_poses=ee_poses,
            allow_grasp=True, 
            check_grasp_ids=self.inhand,
            check_relative_pose=self.check_relative_pose,
        ) # qpos for the robots only

        # 현재 arm의 qpos를 IK를 풀어나온 target_qpos로 변경함.
        for _name, ik_result in qpos_target_dict.items():
            if ik_result is not None:
                robot_qpos, qpos_idxs = ik_result[0], ik_result[1] 
                full_qpos_result[qpos_idxs] = robot_qpos

        # (1) arm_joint target_qpos (2)현재 환경 qpos 중 arm_joint qpos만 target_qpos 바꾼 것  
        return qpos_target_dict, full_qpos_result

    def parse_llm_plan_to_qpos(
        self, 
        physics, 
        path_plan: LLMPathPlan, 
        verbose: bool = False,
        update: bool = False,
    ) -> Tuple[np.ndarray]:
        """
        Assumes the paths for each robot are the same length, and some might end with a grasp/release.
        returns the target qpos after computing IK on all the goal/waypoint poses 
        """

        # (1) path_plan.ee_target_poses의 IK를 풀어서 나온 arm_joint의 target_qpos
        # (2) path_plan.ee_target_poses의 IK를 풀어서 나온 physics내의 모든 model의 qpos
        qpos_target_dict, full_qpos_target = self.ik_ee_poses_to_qpos(
            physics, path_plan.ee_target_poses
        )

        for _name, ik_result in qpos_target_dict.items():
            assert ik_result is not None, f"failed to compute IK for {_name}" 

        # 협동 task에 참여하는 로봇들의 target_joint qpos를 return.  == qpos_target_dict
        joint_qpos_target = full_qpos_target[self.rrt_planner.all_joint_idxs_in_qpos]

        # target_qpos are NOT allowed to be IK-insolvable, but waypoints might be
        ee_waypoints_list = path_plan.ee_waypoints_list 

        # 1. 각 waypoint를 target_pose로 IK를 풀어서 arm_joint의 target_qpos들을 구함
        # 2. physics내의 모든 model의 qpos중 arm_joint부분을 (1)로 대체.
        # 3. waypoints_full_qpos에 모음.
        waypoints_full_qpos = []
        for tstep, ee_poses in enumerate(ee_waypoints_list):  
            attempt_qpos_dict, attempt_full_qpos = self.ik_ee_poses_to_qpos(
                physics, ee_poses
                ) 

            if all([ik_result is not None for ik_result in attempt_qpos_dict.values()]): 
                waypoints_full_qpos.append(attempt_full_qpos)


        print(f"Given {len(ee_waypoints_list)} waypoints, found {len(waypoints_full_qpos)} valid waypoints via IK")
        # physics내의 모든 model의 qpos중에서 로봇들의 arm_joint에 해당하는 qpos만을 인덱싱해서 모음.
        joints_qpos_waypoints = [
            qpos[self.rrt_planner.all_joint_idxs_in_qpos] for qpos in waypoints_full_qpos
            ]
        if verbose:
            print(f"found {len(waypoints_full_qpos)} valid waypoints via IK")

        # Policy(=self)를 target으로 변화시킨 qpos들로 업데이트
        if update: # True
            self.full_qpos_target = full_qpos_target
            self.joints_qpos_target = joint_qpos_target
            self.waypoints_full_qpos = waypoints_full_qpos
            self.joints_qpos_waypoints = joints_qpos_waypoints

        return full_qpos_target, waypoints_full_qpos, joint_qpos_target, joints_qpos_waypoints

    def parse_llm_plan_for_grasp(self, physics, path_plan: LLMPathPlan) -> Dict[str, Tuple[str, str, int]]:
        """ parses each object to grasp/release """
        tograsp = dict()

        for robot_name, obj in path_plan.tograsp.items():
            tograsp[robot_name] = None 
            if obj is not None:
                # make sure the object is in the physics
                obj_name, obj_site_name = obj[0], obj[1]
                grasp = obj[2] # 1 or 0
                if 'rope' in obj_name:
                    weld_body_name = self.robots[robot_name].weld_body_name
                    # special case for rope task 
                    if 'front' in obj_name: 
                        weld_name = f'rope_front_end_{weld_body_name}'    
                        body_name = 'CB0'

                    elif 'back' in obj_name:
                        weld_name = f'rope_back_end_{weld_body_name}'
                        body_name = 'CB24'

                    else:
                        print(obj_name)
                        breakpoint()
                    weld_id = physics.named.model.eq_active._convert_key(weld_name)
                    tograsp[robot_name] = dict(
                        obj_name=obj_name,
                        grasp_site_name=body_name,
                        grasp_val=grasp,
                        weld_id=weld_id,
                        weld_name=weld_name,
                        )
                    continue

                try:
                    obj_body = physics.model.body(obj_name)
                except:
                    raise ValueError(f"object {obj_name} not in physics")
                try:
                    obj_site = physics.data.site(obj_site_name)
                except:
                    raise ValueError(f"object site {obj_site} not in physics")
                
                # (1) 잡을 object의 이름, (2) 잡을 위치(site), (3) 잡는 지 여부
                tograsp[robot_name] = dict(
                    obj_name=obj_name, 
                    grasp_site_name=obj_site_name, 
                    grasp_val=grasp,
                    )
                if self.use_weld:
                    weld_body_name = self.robots[robot_name].weld_body_name
                    weld_name = f"{obj_site_name}_{weld_body_name}" # e.g. apple_top_rhand
                    try:
                        enabled = physics.named.model.eq_active[weld_name] 
                        weld_id = physics.named.model.eq_active._convert_key(weld_name)
                        tograsp[robot_name]["weld_id"] = weld_id # change to weld id!
                        tograsp[robot_name]["weld_name"] = weld_name
                    except KeyError:
                        print(f"{weld_name} not found in eq_active")
                        breakpoint()
                        continue

        # 잡을 물체의 정보에 대한 파싱 (1) 잡을 object의 이름, (2) 잡을 위치(site), (3) 잡는 지 여부         
        return tograsp 

    def plan_qpos(self, physics):
        start_qpos = physics.data.qpos.copy() # 현재시점 qpos
        joints_start_qpos = start_qpos[self.rrt_planner.all_joint_idxs_in_qpos] # 로봇팔 joint의 현재시점 qpos
        
        # physics_cp = physics.copy(share_model=True)
        # for qpos in self.joints_qpos_waypoints + [self.joints_qpos_target]:
        #     # physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos] = qpos
        #     physics_cp = self.rrt_planner.forward_kinematics_all(
        #        q=qpos, physics=physics_cp, return_ee_pose=False,
        #     )
            
        #     img_arr = physics_cp.render(
        #         camera_id='teaser', height=400, width=600,
        #     )
        #     physics_cp.data.qpos[:] = start_qpos
        #     physics_cp.forward()
        #     img_arr = np.concatenate([img_arr, physics_cp.render(
        #         camera_id='teaser', height=400, width=600,
        #         )], axis=1)
        #     plt.imshow(img_arr)
        #     plt.show() 
        #     # dist = np.linalg.norm(
        #     # physics_cp.data.site('robotiq_ee').xpos - physics_cp.data.site('obstacle_wall_front_top').xpos
        #     # )
        #     # print(f"dist: {dist}")
        #     # print(physics_cp.model.contacts)
        # breakpoint()

        plan_fn = self.rrt_planner.plan # plan
        if self.plan_splitted:
            plan_fn = self.rrt_planner.plan_splitted
        
        # list= [로봇팔의 현재시점qpos부터 target_qpos까지 Path, PathPlanning성공텍스트]
        path = plan_fn(
            start_qpos=joints_start_qpos,
            goal_qpos=self.joints_qpos_target,
            skip_endpoint_collision_check=0,
            init_samples=self.joints_qpos_waypoints[::-1], # NOTE: reverse waypoints
            allow_grasp=True, 
            check_grasp_ids=self.grasp_allowed,
            skip_direct_path=self.skip_direct_path,
            skip_smooth_path=self.skip_smooth_path,
            check_relative_pose=self.check_relative_pose,
            timeout=self.timeout,
        )

        # 길찾기 실패
        if path[0] is None:
            print(f"failed to find a path, reason: {path[1]}")
            physics_cp = physics.copy(share_model=True)
            physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos]  = self.joints_qpos_target
            qpos_str = " ".join(physics_cp.data.qpos.astype(str))
            print(f"<key name='rrt_fail' qpos='{qpos_str}'/>")
            # physics_cp.forward()
            # img_arr = physics_cp.render(
            # camera_id='teaser', height=400, width=400,
            # )
            # physics_cp.data.qpos[:] = start_qpos
            # physics_cp.forward()
            # img_arr = np.concatenate([img_arr, physics_cp.render(
            #     camera_id='teaser', height=400, width=600,
            #     )], axis=1)
            # plt.imshow(img_arr)
            # plt.show()
            # breakpoint()
            return None, path[1]
        
        # 길찾기 성공.
        path_ls = list(path[0]) # path
        path_ls = path_ls[::self.control_freq] + path_ls[-3:-1] # 설정한 주기로 path를 샘플링(마지막은 세밀하게)
        return path_ls, path[1]
    
    def map_qpos_to_ctrl(self, physics, qpos: np.ndarray, include_inhand: bool = True) -> Dict[str, np.ndarray]:

        ctrl_idxs = []
        qpos_idxs = []
        # control이 필요한 모든 joint들을 모음.
        for robot_name, robot in self.robots.items():
            # _vals, _idxs = robot.map_qpos_to_joint_ctrl(qpos)
            ctrl_idxs.extend(robot.joint_idxs_in_ctrl)

        assert len(ctrl_idxs) == len(qpos), "qpos and ctrl do not match"

        # 단순히 각 joint의 제어값을 qpos로 할당함.
        ctrl_vals = qpos.copy().tolist()
        qpos_target = qpos.copy()

        for robot_name, robot in self.robots.items():
            if include_inhand and len(self.inhand[robot_name]) > 0:
                # robot should keep grasping the object
                grasp_ctrl_val = robot.get_grasp_ctrl_val(grasp=1) # single number
                ctrl_vals.append(grasp_ctrl_val)
                ctrl_idxs.append(robot.grasp_idx)
                # print(f"robot {robot_name} keeps grasping {self.inhand[robot_name]}: ctrl_val={grasp_ctrl_val} ctrl_idx={robot.grasp_idx}")

            qpos_idxs.extend(
                robot.joint_idxs_in_qpos
            ) 

        # 한 step의 qpos에 대한 정보를 모아서 return.
        return dict(
            ctrl_idxs=np.array(ctrl_idxs),
            ctrl_vals=np.array(ctrl_vals),
            # NOTE: setting qpos-target makes motion jitter a lot
            qpos_idxs=np.array(qpos_idxs),
            qpos_target=qpos_target,
            )

    def get_grasp_action(
        self,
        physics, 
        qpos,
    ) -> List[SimAction]:
        '''
        최종qpos의 action_signal을 추론
        '''
        # 마지막 qpos -> action을 위한 정보로 변환
        joint_ctrls = self.map_qpos_to_ctrl(physics, qpos)
        eq_active_idxs = []
        eq_active_vals = []

        # 마지막 qpos의 FK을 계산하여 target_EE_6d-pose 추론
        target_ee_poses = self.rrt_planner.forward_kinematics_all(
            physics=physics.copy(share_model=True),
            q=qpos,
            return_ee_pose=True,
        )
        grasp_idxs, grasp_vals = [], []
        for robot_name, obj_info in self.tograsp.items(): 
            if obj_info is not None: 
                grasp_val = obj_info["grasp_val"]
                if 'rope' in obj_info['obj_name']:  
                    grasp_vals.append(
                        self.robots[robot_name].get_grasp_ctrl_val(grasp=(grasp_val > 0))
                        )
                    grasp_idxs.append(
                        self.robots[robot_name].grasp_idx
                    )
                    if self.use_weld and obj_info.get("weld_id", None) is not None:
                        # both adhesion and eq_active is turned on
                        weld_id = obj_info["weld_id"]
                        weld_name = obj_info["weld_name"]
                        eq_active_idxs.append(weld_id)
                        assert int(grasp_val) in [int(0), int(1)], f"grasp_val should be integer 0 or 1 when using weld"
                        eq_active_vals.append(int(grasp_val))
                    continue 

                obj_site = obj_info["grasp_site_name"] # 물체의 잡을 부분(site) return.
                site_xpos = physics.data.site(obj_site).xpos # 물체의 잡을 부분의 카타시안좌표.
                if grasp_val > 0:
                    pose = target_ee_poses[robot_name] # 각 로봇팔의 목표_EE_6D_Pose
                    robot_ee_pos = pose.position # 각 로봇팔의 목표_EE_3d_position
                    dist = np.linalg.norm(site_xpos - robot_ee_pos) # 잡을 위치_position과 현재EE_postion 사이 거리.
                    if dist > 0.1:
                        print(f"WARNING: robot {robot_name} end effector distance: {dist} is too far from object {obj_info['obj_name']}")   
                
                grasp_idxs.append(
                    self.robots[robot_name].grasp_idx
                )

                grasp_ctrl_val = self.robots[robot_name].get_grasp_ctrl_val(grasp=(grasp_val > 0))
                grasp_vals.append(grasp_ctrl_val)
                # print(f'seting grasp of robot {robot_name} to {grasp_val}')
                if self.use_weld and obj_info.get("weld_id", None) is not None:
                    # both adhesion and eq_active is turned on
                    weld_id = obj_info["weld_id"]
                    weld_name = obj_info["weld_name"]
                    eq_active_idxs.append(weld_id)
                    assert int(grasp_val) in [int(0), int(1)], f"grasp_val should be integer 0 or 1 when using weld"
                    eq_active_vals.append(int(grasp_val))
        
        if len(grasp_idxs) > 0:
            joint_ctrls["ctrl_idxs"] = np.concatenate(
                [joint_ctrls["ctrl_idxs"], np.array(grasp_idxs)]
            )
            joint_ctrls["ctrl_vals"] = np.concatenate(
                [joint_ctrls["ctrl_vals"], np.array(grasp_vals)]
            )
            joint_ctrls['eq_active_idxs'] = np.array(eq_active_idxs)
            joint_ctrls['eq_active_vals'] = np.array(eq_active_vals)

        # joint_control_signal return.
        return [SimAction(**joint_ctrls)]
    
    def plan_home(
        self,
        physics, 
        start_qpos,
    ) -> List[SimAction]:
        # TODO: try arm returns home but base stays fixed?
        
        need_plan = False
        home_qpos = self.full_qpos_target.copy() # 
        all_qpos_idxs = self.rrt_planner.all_joint_idxs_in_qpos # 모든 로봇팔들의 조인트
        home_qpos[all_qpos_idxs] = start_qpos
        
        for agent_name, return_home in self.path_plan.return_home.items():
            if return_home:
                need_plan = True
                qpos_idxs = self.robots[agent_name].joint_idxs_in_qpos
                robot_qpos = self.robots[agent_name].get_home_qpos()
                home_qpos[qpos_idxs] = robot_qpos
        
        if not need_plan:
            return []
        
        goal_qpos = home_qpos[all_qpos_idxs]
        # TODO: handle return home with object already dropped
        path = self.rrt_planner.plan(
            start_qpos=start_qpos,
            goal_qpos=goal_qpos,
            skip_endpoint_collision_check=1,
            init_samples=[],
            allow_grasp=True, 
            check_grasp_ids=self.grasp_allowed,
            skip_direct_path=self.skip_direct_path,
            skip_smooth_path=self.skip_smooth_path,
            check_relative_pose=self.check_relative_pose,
        )
        if path[0] is None:
            print(f"Failed to find a path to return to Home, reason: {path[1]}")
            physics_cp = physics.copy(share_model=True)
            physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos]  = goal_qpos
            qpos_str = " ".join(physics_cp.data.qpos.astype(str))
            print(f"<key name='rrt_return_home_fail' qpos='{qpos_str}'/>")
            # breakpoint()
            return []
        else:
            
            print(f"Found a path to return to Home")
            path_ls = list(path[0])
            path_ls = path_ls[::self.control_freq] + path_ls[-3:-1]
            actions = []
            for qpos in path_ls:
                kwargs = self.map_qpos_to_ctrl(physics, qpos, include_inhand=False) # avoid gripper keep grasping after placing
                actions.append(SimAction(**kwargs))
            
            return actions
        
                
    def plan(self, env) -> bool: 
        """
        plan a series of actions for each robot
        """  
        physics = env.physics

        # 시작qpos부터 목표qpos까지 path_plan을 return.
        path_ls, reason = self.plan_qpos(physics)
        self.rrt_plan_results = path_ls 
        if path_ls is None:
            return False, reason
        
        # path_plan -> action_signal 맵핑.
        actions = []
        for qpos in path_ls:
            # qpos를 control로 맵핑하기 위한 정보들 return.
            kwargs = self.map_qpos_to_ctrl(physics, qpos)
            # qpos를 control_signal로 맵핑
            actions.append(SimAction(**kwargs))

        end_qpos = path_ls[-1]
        # 마지막 qpos를 통해서 grasping action을 추출.-> 잡기(255), 열기(0) 
        grasp_actions = self.get_grasp_action(physics, end_qpos)
        actions.extend(grasp_actions) 
        actions.extend(
            self.plan_home(physics, end_qpos)
        )
        # 모든 path-plan을 액션으로 변환한 것을 self.action_buffer이 저장.  
        self.action_buffer = actions  
        self.action_idx = 0

        return True, reason
    
    @property
    def plan_exhausted(self) -> bool:
        return self.action_idx == len(self.action_buffer)
    
    @property
    def num_actions(self) -> int:
        return len(self.action_buffer)

    def act(self, obs: EnvState, physics) -> SimAction:
        if self.close_loop:
            replanned, reason = self.plan(physics)
            if not replanned:
                print("replanning failed, using previous plan")
        else:
            assert len(self.action_buffer) != 0, "action buffer is empty, cal plan_qpos first"
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        return action 
 