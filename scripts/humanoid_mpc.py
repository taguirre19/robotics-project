import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from dataclasses import dataclass


@dataclass
class foot:
    x: float
    y: float

@dataclass
class feet:
    spread: float
    length: float
    width: float
    right: foot
    left: foot



class MPCParams():
    def __init__(self, T, N, h_CoM, g, robot_feet, duration=8, step_duration=1):
        self.T = T
        self.duration = duration
        self.N = N
        self.h_CoM = h_CoM
        self.g = g
        self.Px = np.array([[1., (i+1)*T, (((i+1)*T)**2)/2 - h_CoM/g] for i in range(self.N)])
        self.Pu = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.Pu[i, i-j] = (1 + 3*j + 3*(j**2)) * (T**3/6) - T*h_CoM/g

        self.A = np.array([[1., T, (T**2)/2], [0., 1., T], [0., 0., 1.]]) 
        self.b = np.array([(T**3)/6., T**2/2., T])

        self.e = np.array([1., 0., h_CoM/g])
        
        self.x = np.zeros(3)
        self.y = np.zeros(3)

        self.P = np.identity(self.N)
        self.q = np.zeros(self.N)

        self.feet_tracker = np.ones((int(self.duration/self.T), 2))

        self.Zmin, self.Zmax = self.generate_foot_trajectory(robot_feet, step_duration)
    def compute_next_coord(self, coord):
        if coord == 'x':        
            return self.A @ self.x + self.jerk * self.b
        elif coord == 'y':
            return self.A @ self.y + self.jerk * self.b
        else:
            raise ValueError('coord should be x or y')
        
    def solve_step_k(self,
                    Zmin_k: np.array, 
                    Zmax_k: np.array,
                    solver: str,
                    coord: str='x'):
        pass

    
    def solve(self, Zmin, Zmax, coord, solver='daqp'):
        jerks = []
        coord_path = []
        z_path = []
        timesteps = int(self.duration/self.T)
        for k in range(timesteps-self.N):
            jerk = self.solve_step_k(Zmin[1+k:1+k+self.N], Zmax[1+k:1+k+self.N], solver, coord)
            if jerk is None:
                break

            self.jerk = jerk[0]
            if coord == 'x':
                self.x = self.compute_next_coord('x')
                self.z = self.e @ self.x
                coord_path.append(self.x[0])
            elif coord == 'y':
                self.y = self.compute_next_coord('y')
                self.z = self.e @ self.y
                coord_path.append(self.y[0])

            jerks.append(self.jerk)            
            z_path.append(self.z)

        return coord_path, z_path, jerks


    def generate_foot_trajectory(self, robot_feet, step_duration):
        timesteps = int(self.duration/self.T)
        nb_steps = int(self.duration/step_duration) - 2
        nb_samples_per_step = int(step_duration/self.T)

        Zxmin = np.zeros(timesteps)
        Zxmax = np.zeros(timesteps)
        Zymin = np.zeros(timesteps)
        Zymax = np.zeros(timesteps)
        feet_tracker = np.ones((timesteps, 2), dtype=int)

        # First step both robot_feet are at the ground
        Zxmin[:nb_samples_per_step] = robot_feet.right.x - robot_feet.width/2
        Zxmax[:nb_samples_per_step] = robot_feet.left.x + robot_feet.width/2
        Zymin[:nb_samples_per_step] = robot_feet.right.y - robot_feet.length/2
        Zymax[:nb_samples_per_step] = robot_feet.right.y + robot_feet.length/2

        LEFT = 0
        RIGHT = 1

        foot = RIGHT

        current = nb_samples_per_step
        for step in range(nb_steps):
            if foot == LEFT:
                Zxmin[current:current+nb_samples_per_step] = robot_feet.left.x - robot_feet.width/2
                Zxmax[current:current+nb_samples_per_step] = robot_feet.left.x + robot_feet.width/2

                foot = RIGHT
                feet_tracker[current:current+nb_samples_per_step, 1] = 0
            else:
                Zxmin[current:current+nb_samples_per_step] = robot_feet.right.x - robot_feet.width/2
                Zxmax[current:current+nb_samples_per_step] = robot_feet.right.x + robot_feet.width/2
                foot = LEFT
                feet_tracker[current:current+nb_samples_per_step, 0] = 0


            Zymin[current:current+nb_samples_per_step] = robot_feet.left.y - robot_feet.length/2 + (step+1) * robot_feet.spread
            Zymax[current:current+nb_samples_per_step] = robot_feet.left.y + robot_feet.length/2 + (step+1) * robot_feet.spread

            current += nb_samples_per_step

        # Last step both robot_feet are at the ground
        Zxmin[current:] = robot_feet.right.x - robot_feet.width/2
        Zxmax[current:] = robot_feet.left.x + robot_feet.width/2
        Zymin[current:] = robot_feet.left.y - robot_feet.length/2 + (nb_steps+1) * robot_feet.spread
        Zymax[current:] = robot_feet.left.y + robot_feet.length/2 + (nb_steps+1) * robot_feet.spread


        # Stack in one np array
        Zmin = np.vstack((Zxmin, Zymin))
        Zmax = np.vstack((Zxmax, Zymax))

        self.feet_tracker = feet_tracker

        return Zmin, Zmax


class MPCClassic(MPCParams):
    def solve_step_k(self,
            Zmin_k: np.array, 
            Zmax_k: np.array,
            solver: str,
            coord: str='x'):
        beta = 2
        alpha = 10e-6 *  beta

        Zref = (Zmax_k - Zmin_k) / 2
        self.P = alpha * np.identity(self.N) + beta * self.Pu.T @ self.Pu
        self.q = beta * self.Pu.T @ (self.Px @ self.x - Zref)
        G = np.zeros((2 * self.N, self.N))
        h = np.zeros((1, 2*self.N))

        jerk = solve_qp(self.P, self.q, G=G, h=h, solver=solver)

        return jerk

            

class MPCRobust(MPCParams):
    def solve_step_k(self,
                Zmin_k: np.array, 
                Zmax_k: np.array,
                solver: str,
                coord: str='x'):
        G = np.vstack((self.Pu, -self.Pu))
        if coord == 'x':
            h = np.hstack((Zmax_k- self.Px @ self.x, self.Px @ self.x -Zmin_k))
        elif coord == 'y':
            h = np.hstack((Zmax_k- self.Px @ self.y, self.Px @ self.y -Zmin_k))

        jerk = solve_qp(self.P, self.q, G=G, h=h, solver=solver)

        return jerk


class MPCForce(MPCRobust):
    def compute_next_coord(self, coord):
        force = 55e-5
        if coord == 'x':
            return self.A @ self.x + self.jerk * self.b + force
        elif coord == 'y':
            return self.A @ self.y + self.jerk * self.b
        raise ValueError('coord should be x or y')