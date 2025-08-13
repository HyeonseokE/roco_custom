import os
import copy
import time
import cv2 
import random
import numpy as np  
from pydantic import dataclasses, validator 
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control 
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, PANDA_CONSTANTS

PACK_TASK_OBJECTS=[
    "bin",
    "table_top",
    "apple",
    "banana",
    "milk",
    "soda_can",
    "bread",
    "cereal",
]
PACK_ITEM_NAMES=[
    "apple",
    "banana",
    "milk",
    "soda_can",
    "bread",
    "cereal",
]
PACK_BIN_SITE_NAMES=[
    "bin_front_left",
    "bin_front_right",
    "bin_front_middle",
    "bin_back_left",
    "bin_back_right", 
    "bin_back_middle",
]
 
PACK_TASK_CONTEXT="""[Task Description]
Two robots, Alice and Bob, each stands at a different side of the table, and together pack all the grocery items on the table into a bin.
They choose objects closest to their grippers. At each round, they are given [Scene description], [Environment feedback], and must reason about the task. Each robot does **exactly** one ACTION and PATH per round, their PATHs must avoid collision.
"""

PACK_ACTION_SPACE="""
[Action Options]
1) PICK <obj> PATH <path>: only PICK if your gripper is empty;
2) PLACE <obj> bin PATH <path>: only if you have already PICKed the object, you can PLACE it into an empty bin slot, do NOT PLACE if another object is already in a slot!

Each <path> must contain exactly four <coord>s that smoothly interpolate between start and goal, coordinates must be evenly distanced from each other.
The robot PATHs must efficiently reach target while avoiding collision avoid collision (e.g. move above the objects' heights).
The PATHs must do top-down pick or place: 
- move directly atop an object by height 0.2 before PICK: e.g. Alice's gripper is at (0, 0, 0.3), banana is at (-0.25, 0.39, 0.29): NAME Alice ACTION PICK banana PATH [(0, 0.1, 0.3),(0, 0.2, 0.49),(-0.1, 0.25, 0.49),(-0.25, 0.39, 0.49)]
- lift an object vertically up before moving it to PLACE: e.g. Bob's gripper is at (0.9, 0, 0.2), bin_front_left is at (0.35, 0.35, 0.43): NAME Bob ACTION PLACE apple bin_front_left PATH [(0.9,0.0,0.5), (0.5, 0, 0.5), (0.2, 0.1, 0.5),(0.35, 0.35, 0.5)]

[Action Output Instruction]
First output 'EXECUTE\n', then give exactly one ACTION per robot, each on a new line.
Example: 'EXECUTE\nNAME Alice ACTION PICK apple PATH <path>\nNAME Bob ACTION PLACE banana bin_back_middle PATH <path>\n'
"""

PACK_CHAT_PROMPT="""Robots discuss to find the best strategy and path. When each robot talk, it first reflects on the task status and its own capability. 
Carefully consider [Environment Feedback]. Coordinate with others to plan and improve paths following the instructions. They talk in order [Alice],[Bob],[Alice],..., then, after they agreed, plan exactly one ACTION per robot, output an EXECUTE to summarize the plan and stop talking.
Their discussion and the final plan: """

class PackGroceryTask(MujocoSimEnv):
    def __init__( 
        self,
        filepath: str = "rocobench/envs/task_pack.xml",
        one_obj_each: bool = False,
        **kwargs,
    ):    
        self.robot_names = ["ur5e_robotiq", "panda"] 
        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob", 
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda", 
        }
        self.robots = dict()  

        # simulation 환경 속 로봇 설정 들
        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()  
        panda_config = PANDA_CONSTANTS.copy() 

        self.item_names = PACK_ITEM_NAMES

        # parent-class init -> self.physics
        super(PackGroceryTask, self).__init__(
            filepath=filepath,  
            task_objects=PACK_TASK_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=robotiq_config,
                panda=panda_config, 
            ),
            **kwargs
        ) 

        # packing task destination_location init
        self.bin_slot_xposes = dict()
        for sname in PACK_BIN_SITE_NAMES:
            self.bin_slot_xposes[sname] = self.physics.data.site(sname).xpos.copy()

        # robot embodiment init
        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )
         
        self.align_threshold = 0.06
    
    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]: 
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]

        if target_name in self.item_names:
            sname = f"{target_name}_top"  
        elif target_name in self.bin_slot_xposes.keys():
            sname = target_name
        else:
            return None 
        try:
            ret = self.physics.data.site(sname).xpos.copy() 
        except KeyError:
            print(f"KeyError: {sname} not in model sites")
            pass

        return ret

    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if target_name in self.item_names:
            sname = f"{target_name}_top" 
        elif target_name in self.bin_slot_xposes.keys():
            sname = target_name
        else:
            return None 
        try:
            ret = self.physics.data.site(sname).xmat.copy().reshape(3, 3)
            ret = mat_to_quat(ret)
            if any([name in sname for name in ['apple', 'soda_can', 'milk']]):
                # change quat
                if agent_name == "Bob":
                    ret = np.array([1, 0, 0, 1])
                else:
                    ret = np.array([1, 0, 0, 0])
            if 'bin_' in target_name and agent_name == "Bob":
                ret = np.array([1, 0, 0, 1])
        except KeyError:
            print(f"KeyError: {sname} not in model sites")
            pass
        return ret 
    
    @property 
    def use_prepick(self):
        return False  

    @property
    def use_preplace(self):
        return False
    
    @property
    def waypoint_std_threshold(self):
        return 0.19

    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        '''
        책상과 바구니의 바닥면과 모든 물체 사이의 collision_pair(충돌허용)를 return
        '''
        bin_id = self.physics.model.body("bin").id
        bin_bottom_id = self.physics.model.body("bin_inside").id
        table_id = self.physics.model.body("table").id

        ret = [(table_id, bin_bottom_id)]
        all_body_ids = []
        for obj_name in self.item_names:
            body_ids = self.get_all_body_ids(obj_name)
            for body_id in body_ids:
                ret.append((body_id, bin_bottom_id))
                # ret.append((body_id, bin_id)) this makes direct path less likely
                ret.append((body_id, table_id))
                all_body_ids.append(body_id)

        ee_link_ids = self.robots["Alice"].ee_link_body_ids + self.robots["Bob"].ee_link_body_ids
        ee_link_ids = [_id for _id in ee_link_ids if _id != "panda_hand"]

        return ret 

    def get_graspable_objects(self):
        graspables = self.item_names.copy()
        return dict(
            Alice=graspables,
            Bob=graspables, 
        )

    def get_grasp_site(self, obj_name: str = "apple") -> Optional[str]:
        if obj_name in self.item_names:
            return f"{obj_name}_top"
        else:
            return None

    def get_object_joint_name(self, obj_name: str) -> str:
        return f"{obj_name}_joint"

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name] 

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.3, 1.6), y=(-0.4, 1.5), z=(0, 1))
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-1.3, 1.6), y=(0, 1.5), z=(0, 1))
        else:
            raise NotImplementedError
    
    def sample_initial_scene(self): 
        tosample_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'grid' in geom.name:
                low = geom.pos - geom.size
                high = geom.pos + geom.size
                tosample_panels.append(
                    (low, high)
                )
        assert len(tosample_panels) >= len(self.item_names), "Not enough grid positions to sample from"
        panel_idxs = self.random_state.choice(
            len(tosample_panels), 
            len(self.item_names),
            replace=False
            )
        for _idx, item_name in zip(panel_idxs, self.item_names):
            low, high = tosample_panels[_idx]
            new_pos = self.random_state.uniform(low, high) 
            new_pos[2] = self.physics.data.body(item_name).xpos[2] # height stays same!
            new_quat = Quaternion(
                axis=[0,0,1], 
                angle=self.random_state.uniform(low=0, high=2*np.pi)
                ) 
            new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
            self.reset_body_pose(
                body_name=item_name,
                pos=new_pos,
                quat=new_quat,
            )  
            self.reset_qpos(
                jnt_name=f"{item_name}_joint",
                pos=new_pos,
                quat=new_quat,
            )
          
        self.physics.forward()
        self.physics.step(50)
    
    def get_obs(self) -> EnvState:

        # task_env 시나리오 안에서 다른 객체와 contact되어있는 모든 객체를 return.
        contacts = self.get_contact()

        # grasping이 허용된 객체를 로봇이 contact하고 있다면, 각 로봇의 contact에 포함시킴.
        allow_objs = self.item_names + ["bin", "table"]
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in allow_objs]
        contacts["panda"] = [c for c in contacts["panda"] if c in allow_objs]

        # 각 grasable객체들의 states를 얻음.
        obj_states = self.get_object_states(contact_dict=contacts)
        agent_states = dict()

        # task_env에 참여하는 robot_agent의 states를 얻음.
        for agent_name, agent_constants in self.agent_configs.items():
            agent_state = self.get_agent_state(
                agent_constants, contact_dict=contacts
            ) 
            agent_states[agent_name] = agent_state

        # graspable객체들과 로봇들의 states를 모음.
        kwargs = dict(objects=obj_states,)
        kwargs.update(agent_states)


        if self.render_point_cloud:
            point_cloud = self.get_point_cloud()
            kwargs['scene'] = point_cloud # NOTE: should include bboxes! 
        obs = EnvState(**kwargs)
         
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation" 
        return obs
    
    def get_reward_done(self, obs): 
        all_packed = True
        reward = 1
        for food in self.item_names:
            bin_coord = self.physics.data.body("bin").xpos[:2]
            dist = np.linalg.norm(obs.objects[food].xpos[:2] - bin_coord)
            if 'bin_inside' not in obs.objects[food].contacts and dist > self.align_threshold:
                all_packed = False 
                reward = 0
                break 
        return reward, all_packed

    def get_contact(self):

        # contact이 되어있는 모든 객체들을 모아서 return
        contacts = super().get_contact() # 

        # (ur5e_robotiq 로봇바디를 제외하고)로봇 부품과 contact된 객체가 있다면, 그것들만 contact[ur5e_robotiq]에 유지.
        robotiq_link_names = self.agent_configs["ur5e_robotiq"]['all_link_names'] + ['ur5e_robotiq']
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c not in robotiq_link_names] 

        # (panda 로봇바디를 제외하고)로봇 부품과 contact된 객체가 있다면, 그것들만 contact[panda]에 유지.
        panda_link_names = self.agent_configs["panda"]['all_link_names'] + ["panda_right_finger", "panda_left_finger", "panda"]
        contacts["panda"] = [c for c in contacts['panda'] if c not in panda_link_names] 
        contacts["panda"].append("broom")

        return contacts

    def central_plan_prompt(self, chat_history: List[str] = []):
        return PACK_PLAN_PROMPT 

    def get_action_prompt(self) -> str:
        
        return PACK_ACTION_SPACE

    def describe_object(self, obs, name):
        '''
        특정 객체와 contact와의 상호작용에 대한 프롬프트를 생성 후 return.
        ex) name = "apple"
            contacts = "table"
            -> object_desp = "apple: (-0.54, 0.58, 0.30) on the table"
        '''
        x,y,z = self.physics.data.site(f"{name}_top").xpos # {name}객체의 xpos
        z += 0.05 # further avoid collision
        contacts = obs.objects[name].contacts # {name}객체와 contact된 객체 return
        object_desp = f"{name}: ({x:.2f}, {y:.2f}, {z:.2f}), "
        if 'bin_inside' in contacts:
            dist_to_slot = [
                (
                    slot_name, np.linalg.norm(np.array([x,y]) - slot_xpos[:2])
                ) for slot_name, slot_xpos in self.bin_slot_xposes.items()

            ]
            slot_name = min(dist_to_slot, key=lambda x: x[1])[0]
            object_desp += f"inside slot {slot_name}"
        else:
            object_desp += f"on table"
        
        #
        return object_desp

    def describe_robot_state(self, obs, robot_name):
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        contacts = robot_state.contacts 
        contacts = [c for c in contacts if c in self.item_names]
        obj = contacts[0] if len(contacts) > 0 else "nothing"
        agent_name = self.robot_name_map[robot_name]
        robot_desp = f"{agent_name}'s gripper: ({x:.2f}, {y:.2f}, {z:.2f}), holding {obj}" 
        return robot_desp
    
    def describe_obs(self, obs: EnvState):
        full_desp =  "[Scene description]\n" 
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        full_desp += f"robots must move lower than 0.8 but higher than table height {table_height:.2f}\n"
        for name in self.item_names:
            full_desp += self.describe_object(obs, name) + "\n"

        for slot_name, slot_xpos in self.bin_slot_xposes.items():
            x, y, z = slot_xpos
            full_desp += f"{slot_name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
 
        for robot_name, agent_name in self.robot_name_map.items():
            full_desp += self.describe_robot_state(obs, robot_name) + "\n"
            
        return full_desp 
    
    def describe_task_context(self):
        return PACK_TASK_CONTEXT
    
    def get_agent_prompt(self, obs, agent_name):        

        # 현재 embodiment에 할당된 "이름"과 상대 로봇의 이름을 얻음
        robot_name = self.get_robot_name(agent_name)
        other_robot = "Alice" if agent_name == "Bob" else "Bob"

        # task_env 상에 graspable obeject에 대한 위치(x,y,z), contact정보("on table")등 prompt return
        object_desp = "\n".join([self.describe_object(obs, name) for name in self.item_names])

        table_height = self.physics.data.body("table_top").xpos[2] + 0.15 

        # task_env 상에 gripper 위치(x,y,z), contact정보("hold noting")등 prompt return
        robot_desp = self.describe_robot_state(obs, robot_name).replace(f"{agent_name}'s", "Your")

        # task_env 상에 bin_slot 위치(x,y,z)정보 prompt return
        slot_desp = "\n".join(
            [
                f"{slot_name}: ({x:.2f}, {y:.2f}, {z:.2f})" for slot_name, (x,y,z) in self.bin_slot_xposes.items()
            ]
            )

        # LLM agent에게 먹일 prompt.
        agent_prompt = f"""
You are {agent_name}, you and robot {other_robot} each stands at a different side of the table, and together you must put all the grocery items into a bin.
Locations of slots in the bin:
{slot_desp}
At current round:
You see the following objects:
{object_desp}
{robot_desp}
Your gripper must move higher than these objects and higher than table height {table_height:.2f}, but move lower than 0.8.
Never forget you are {agent_name}!
Think step-by-step about the task and {other_robot}'s response. Carefully check and correct {other_robot} if they made a mistake. 
Discuss with {other_robot} to come up with the best plan and smooth, collision-free paths. 
Improve your paths if given [Environment Feedback], choose a different object or target slot if needed.

When you respond, tell {other_robot} about your status. Respond very concisely but informatively, and do not repeat what others have said.
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction] and [Path Plan Instruction].
"""
        return agent_prompt
    
    def get_task_feedback(self, llm_plan, pose_dict): 
        feedback = ""
        for agent_name, action_str in llm_plan.action_strs.items():
            if 'PICK' not in action_str and 'PLACE' not in action_str:
                feedback += f"{agent_name}'s ACTION is invalid, can only PICK or PLACE"
        return feedback
 
 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image 
    env = PackGroceryTask()
    obs = env.reset()
    print(env.describe_obs(obs))
    print(env.get_agent_prompt(obs, "Alice"))
    print(env.get_agent_prompt(obs, "Bob"))
    breakpoint()
    print(obs.ur5e_robotiq.ee_xquat)
    img=env.physics.render(camera_id="teaser", height=480, width=600)
    im = Image.fromarray(img)
    plt.imshow(img)
    plt.show()
    breakpoint()

