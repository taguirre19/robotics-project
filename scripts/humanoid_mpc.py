import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csr_matrix


class MPCParams():
    def __init__(self, T, N, h_CoM, g, 
                 left_foot_max=0.175, left_foot_min=0.05, 
                 right_foot_max=-0.05, right_foot_min=-0.175, 
                 duration=8, step_duration=1):
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
        
        # initial x
        self.x_dim = 3
        self.x = np.zeros(self.x_dim)

        # output
        self.jerk_dim = self.N

        self.P = np.identity(self.N)
        self.q = np.zeros(self.N)

        self.Zmin, self.Zmax = self.generate_foot_trajectory(left_foot_max, 
                                                             left_foot_min, 
                                                             right_foot_max, 
                                                             right_foot_min, 
                                                             step_duration)
    def compute_next_x(self):        
        return self.A @ self.x + self.jerk * self.b
    
    def solve_step_k(self,
                    Zmin_k: np.array, 
                    Zmax_k: np.array,
                    solver: str):
        G = np.vstack((self.Pu, -self.Pu))
        h = np.hstack((Zmax_k- self.Px @ self.x, self.Px @ self.x -Zmin_k))

        jerk = solve_qp(self.P, self.q, G=G, h=h, solver=solver)

        return jerk

    
    def solve(self, Zmin, Zmax, solver):
        jerks = []
        x_path = []
        z_path = []
        timesteps = int(self.duration/self.T)
        for k in range(timesteps-self.N):
            jerk = self.solve_step_k(Zmin[1+k:1+k+self.N], Zmax[1+k:1+k+self.N], solver)
            if jerk is None:
                break
                # raise ValueError("Not valid solution QP")
            self.jerk = jerk[0]

            self.x = self.compute_next_x()
            self.z = self.e @ self.x
            jerks.append(self.jerk)
            x_path.append(self.x[0])
            z_path.append(self.z)

        return x_path, z_path, jerks

    def generate_foot_trajectory(self, left_foot_max, left_foot_min,
                             right_foot_max, right_foot_min, step_duration):
        timesteps = int(self.duration/self.T)
        nb_steps = int(self.duration/step_duration) - 2
        nb_samples_per_step = int(step_duration/self.T)

        Zmin = np.zeros(timesteps)
        Zmax = np.zeros(timesteps)

        # First step both feet are at the ground
        Zmin[:nb_samples_per_step] = right_foot_min
        Zmax[:nb_samples_per_step] = left_foot_max

        LEFT = 0
        RIGHT = 1

        foot = RIGHT
        current = nb_samples_per_step
        for step in range(nb_steps):
            if foot == LEFT:
                Zmin[current:current+nb_samples_per_step] = left_foot_min
                Zmax[current:current+nb_samples_per_step] = left_foot_max
                foot = RIGHT
            else:
                Zmin[current:current+nb_samples_per_step] = right_foot_min
                Zmax[current:current+nb_samples_per_step] = right_foot_max
                foot = LEFT

            current += nb_samples_per_step

        # Last step both feet are at the ground
        Zmin[current:] = right_foot_min
        Zmax[current:] = left_foot_max

        return Zmin, Zmax
