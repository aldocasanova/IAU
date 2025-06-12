import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time

class KukaArmEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        self.max_steps = 200
        self.step_count = 0
        self.episode_count = 0
        self.contact_made = False
        self.prev_dist = None
        self.cup_initial_pos = None  # para detectar movimiento del vaso

        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.kuka = None
        self.cup_id = None

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.kuka = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        # ✅ Vaso más alejado en el eje X
        if self.episode_count < 100:
            cup_x = 0.6
        else:
            cup_x = np.random.uniform(0.5, 0.65)

        cup_path = os.path.join(os.path.dirname(__file__), "assets/glass.urdf")
        self.cup_initial_pos = [cup_x, 0, 0.06]
        self.cup_id = p.loadURDF(cup_path, basePosition=self.cup_initial_pos)

        # ✅ Posición inicial favorable (doblado hacia adelante)
        initial_angles = [0.0, -0.9, 0.0, -1.4, 0.0, 1.4, 0.0]
        for j in range(7):
            p.resetJointState(self.kuka, j, targetValue=initial_angles[j])

        self.step_count = 0
        self.contact_made = False
        self.episode_count += 1
        self.prev_dist = None

        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.kuka, i)[0] for i in range(7)]
        cup_pos = p.getBasePositionAndOrientation(self.cup_id)[0]
        return np.array(joint_states + list(cup_pos), dtype=np.float32)

    def step(self, action):
        for i in range(7):
            p.setJointMotorControl2(self.kuka, i, p.POSITION_CONTROL,
                                    targetPosition=action[i], force=200)

        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(1./240.)

        efector_pos = p.getLinkState(self.kuka, 6)[0]
        cup_pos, _ = p.getBasePositionAndOrientation(self.cup_id)
        dist = np.linalg.norm(np.array(efector_pos) - np.array(cup_pos))
        efector_height = efector_pos[2]

        terminated = False
        truncated = False
        reward = -0.2 * dist  # penalización por distancia

        # ✅ Bonificación si se acerca al vaso
        if self.prev_dist is not None:
            if dist < self.prev_dist:
                reward += 0.05
            else:
                reward -= 0.05
        self.prev_dist = dist

        # ✅ Penalización si el efector está muy alto
        if efector_height > 0.4:
            reward -= 0.1 * (efector_height - 0.4)

        # ✅ Bonificación si está alineado en eje Y
        y_error = abs(efector_pos[1] - cup_pos[1])
        if y_error < 0.05:
            reward += 0.05

        # ✅ Penalización si el codo (joint 3) está recto; bonificación si doblado
        codo_angle = p.getJointState(self.kuka, 3)[0]
        if abs(codo_angle) < 0.5:
            reward -= 0.1
        elif -1.5 < codo_angle < -0.8:
            reward += 0.05

        # ✅ Golpea el vaso (colisión directa)
        if dist < 0.08:
            p.resetBaseVelocity(self.cup_id, linearVelocity=[0, 0, -1.0])
            reward = 1.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        # ✅ Vaso se movió significativamente (X o Z)
        dx = abs(cup_pos[0] - self.cup_initial_pos[0])
        dz = abs(cup_pos[2] - self.cup_initial_pos[2])
        if dx > 0.02 or dz > 0.02:
            reward = 1.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        # ⏱ Fin por tiempo
        self.step_count += 1
        if self.step_count >= self.max_steps:
            reward = -1.0
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        p.disconnect()
