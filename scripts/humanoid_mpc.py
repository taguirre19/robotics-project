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
    def __init__(self, T, N, h_CoM, g, robot_feet, duration=8, step_duration=1, overlap=None):
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

        self.Zmin, self.Zmax = self.generate_foot_trajectory(robot_feet, step_duration, overlap)
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
        
        G = np.vstack((self.Pu, -self.Pu))
        if coord == 'x':
            h = np.hstack((Zmax_k- self.Px @ self.x, self.Px @ self.x -Zmin_k))
        elif coord == 'y':
            h = np.hstack((Zmax_k- self.Px @ self.y, self.Px @ self.y -Zmin_k))

        jerk = solve_qp(self.P, self.q, G=G, h=h, solver=solver)

        return jerk

    
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


    def generate_foot_trajectory(self, robot_feet, step_duration, overlap=None):
        timesteps = int(self.duration/self.T)
        nb_steps = int(self.duration/step_duration) - 2
        nb_samples_per_step = int(step_duration/self.T)
        if overlap is None:
            overlap = nb_samples_per_step//5

        Zxmin = np.zeros(timesteps)
        Zxmax = np.zeros(timesteps)
        Zymin = np.zeros(timesteps)
        Zymax = np.zeros(timesteps)
        feet_tracker = np.ones((timesteps, 2), dtype=int)
        left_trajectory = []
        right_trajectory = []

        # First step both robot_feet are at the ground
        Zxmin[:nb_samples_per_step] = robot_feet.right.x - robot_feet.width/2
        Zxmax[:nb_samples_per_step] = robot_feet.left.x + robot_feet.width/2
        Zymin[:nb_samples_per_step] = robot_feet.right.y - robot_feet.length/2
        Zymax[:nb_samples_per_step] = robot_feet.right.y + robot_feet.length/2
        left_trajectory.append([robot_feet.left.x, robot_feet.left.y])
        right_trajectory.append([robot_feet.right.x, robot_feet.right.y])

        LEFT = 0
        RIGHT = 1

        foot = RIGHT

        current = nb_samples_per_step
        for step in range(nb_steps):
            if foot == LEFT:
                Zxmin[current:current+nb_samples_per_step-overlap] = robot_feet.left.x - robot_feet.width/2
                Zxmax[current:current+nb_samples_per_step-overlap] = robot_feet.left.x + robot_feet.width/2
                
                Zxmin[current+nb_samples_per_step-overlap:current+nb_samples_per_step] = robot_feet.right.x - robot_feet.width/2
                Zxmax[current+nb_samples_per_step-overlap:current+nb_samples_per_step] = robot_feet.left.x + robot_feet.width/2



                foot = RIGHT
                feet_tracker[current:current+nb_samples_per_step-overlap, 1] = 0
                left_trajectory.append([robot_feet.left.x, robot_feet.left.y + step * robot_feet.spread])


            else:
                Zxmin[current:current+nb_samples_per_step-overlap] = robot_feet.right.x - robot_feet.width/2
                Zxmax[current:current+nb_samples_per_step-overlap] = robot_feet.right.x + robot_feet.width/2

                Zxmin[current+nb_samples_per_step-overlap:current+nb_samples_per_step] = robot_feet.right.x - robot_feet.width/2
                Zxmax[current+nb_samples_per_step-overlap:current+nb_samples_per_step] = robot_feet.left.x + robot_feet.width/2

        


                foot = LEFT
                feet_tracker[current:current+nb_samples_per_step-overlap, 0] = 0
                right_trajectory.append([robot_feet.right.x, robot_feet.right.y + step * robot_feet.spread])


            Zymin[current:current+nb_samples_per_step-overlap] = robot_feet.left.y - robot_feet.length/2 + step * robot_feet.spread
            Zymax[current:current+nb_samples_per_step-overlap] = robot_feet.left.y + robot_feet.length/2 + step * robot_feet.spread
        
            Zymin[current+nb_samples_per_step-overlap:current+nb_samples_per_step] =  robot_feet.left.y - robot_feet.length/2 + step * robot_feet.spread
            Zymax[current+nb_samples_per_step-overlap:current+nb_samples_per_step] =  robot_feet.left.y + robot_feet.length/2 + (step+1) * robot_feet.spread
            current += nb_samples_per_step

        # Last step both robot_feet are at the ground
        Zxmin[current:] = robot_feet.right.x - robot_feet.width/2
        Zxmax[current:] = robot_feet.left.x + robot_feet.width/2
        Zymin[current-overlap:] = robot_feet.left.y - robot_feet.length/2 + (nb_steps-1) * robot_feet.spread
        Zymax[current-overlap:] = robot_feet.left.y + robot_feet.length/2 + (nb_steps-1) * robot_feet.spread

        if foot == LEFT:
            left_trajectory.append([robot_feet.left.x, robot_feet.left.y + (nb_steps-1) * robot_feet.spread])
        else:
            right_trajectory.append([robot_feet.right.x, robot_feet.right.y + (nb_steps-1) * robot_feet.spread])


        # Stack in one np array
        Zmin = np.vstack((Zxmin, Zymin))
        Zmax = np.vstack((Zxmax, Zymax))

        self.feet_tracker = feet_tracker
        self.left_trajectory = left_trajectory
        self.right_trajectory = right_trajectory

        return Zmin, Zmax

class MPCForce(MPCParams):
    def __init__(self, T, N, h_CoM, g, robot_feet, duration=8, step_duration=1, overlap=None, force=10):
        super().__init__(T, N, h_CoM, g, robot_feet, duration, step_duration, overlap)
        self.x = np.zeros(3)
        self.y = np.zeros(3)

        self.P = np.identity(self.N)
        self.q = np.zeros(self.N)
        self.force = force

    # Add force to the problem
    def solve(self, Zmin, Zmax, coord, solver='daqp', force_k=300):
        jerks = []
        coord_path = []
        z_path = []
        timesteps = int(self.duration/self.T)
        for k in range(timesteps-self.N):
            jerk = self.solve_step_k(Zmin[1+k:1+k+self.N], Zmax[1+k:1+k+self.N], solver, coord)
            if jerk is None:
                break

            if k == force_k:
                  self.x[1] += self.force

            if k == force_k + 1:
                self.x[1] -= self.force
      
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



class MPC2Paper:
    def __init__(self, T, N, h_CoM, g, alpha, beta,gamma, m, n_step, n_total):
        self.N = N
        self.h_CoM = h_CoM
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m = m # number of steps
        self.n_total = n_total


        # state variables
        self.x = np.zeros(3)
        self.y = np.zeros(3)
        self.A = np.array([[1., T, (T**2)/2], [0., 1., T], [0., 0., 1.]]) 
        self.b = np.array([(T**3)/6., T**2/2., T])
        self.e = np.array([1., 0., h_CoM/g])
        

        # X_{k+1} = Pps x + Ppu jerk
        # A matrix in paper 1
        self.Pps = np.array([[1, (i+1)*T, (((i+1)*T)**2)/2] for i in range(self.N)]) 
        self.Ppu = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.Ppu[i, i-j] = (1 + 3*j + 3*(j**2)) * (T**3/6)
        
        # X_prima_{k+1} = Pvs x + Pvu jerk
        self.Pvs = np.array([[0, 1, (i+1)*T] for i in range(self.N)]) 
        self.Pvu = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.Pvu[i, i-j] = (1 + 2*j) * (T**2/2)
        
        # Z Cop = Pzs x + Pzu jerk
        # Px and Pu from paper 1
        self.Pzs = np.array([[1., (i+1)*T, (((i+1)*T)**2)/2 - h_CoM/g] for i in range(self.N)])
        self.Pzu = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.Pzu[i, i-j] = (1 + 3*j + 3*(j**2)) * (T**3/6) - T*h_CoM/g

        # X_k, X_prima_k, Z_k
        self.position_x = np.zeros(N)
        self.velocity_x = np.zeros(N)
        self.z_x = np.zeros(N)
        
        self.position_y = np.zeros(N)
        self.velocity_y = np.zeros(N)
        self.z_y = np.zeros(N)
        
    
        # Input vector now:
        # u = [ jerk_x , x_f]
        self.position_f = np.zeros(m)
        self.jerk = np.zeros(N)
        
        # I dont know if we should maintain this
        #self.feet_tracker = np.ones((int(self.duration/self.T), 2))
        self.generator = self.generate_U_Uck(n_total, N, m, n_step)
    
    def generate_U_Uck(self, N_tot, N, m, n):
        for k in range(N_tot):    
            p = k%n     
            Uck = np.zeros((N, 1))
            Uck[:n-p] = 1
            Uk = np.zeros((N, m-1))
            for i in range((N+p)//n):
                Uk[(i+1)*n-p:(i+2)*n-p, i] = 1
            
            yield Uk, Uck

    def update_variables(self, coord):
        if coord == 'x':
            self.x = self.A @ self.x + self.b * self.jerk[0]
            self.position_x = self.Pps @ self.x + self.Ppu @ self.jerk
            self.velocity_x = self.Pvs @ self.x + self.Pvu @ self.jerk
            self.z_x = self.Pzs @ self.x + self.Pzu @ self.jerk
        elif coord == 'y':
            self.x = self.A @ self.y + self.b * self.jerk[0]
            self.position_y = self.Pps @ self.y + self.Ppu @ self.jerk
            self.velocity_y = self.Pvs @ self.y + self.Pvu @ self.jerk
            self.z_y = self.Pzs @ self.y + self.Pzu @ self.jerk
        else:
            raise ValueError('coord should be x or y')
    
    def solve(self, solver, coord):
        # TODO: 
        # - We need a velocity ref
        # - This is only section 3
        # I havent implemented section 4 and 5
        jerks = []
        coord_path = []
        z_path = []
        position_fc = 0
        for k in range(self.n_total - self.N):
            U, Uc = next(self.generator)
            u = self.solve_step_k(Uc, U, position_fc, Vref_value=0.1, solver=solver, coord=coord)
            if u is None:
                raise ValueError
            self.jerk = u[:self.N]
            self.position_f = u[self.N:]
            self.update_variables(coord)
            if coord == 'x':
                coord_path.append(self.x[0])
                z = self.z_x[0] # ? 
            elif coord == 'y':
                coord_path.append(self.y[0])
                z = self.z_y[0] # ?
            jerks.append(self.jerk[0])
            z_path.append(z)
            position_fc = coord_path[-1] 
        return coord_path, z_path, jerks
    
    def solve_step_k(self, Uc, U, position_fc, Vref_value, solver, coord):
        Vref = Vref_value * np.ones(self.N)
        if coord == 'x':
            state_vector = self.x.copy()
        elif coord == 'y':
            state_vector = self.y.copy()
        # TODO: check dimensions
        pk = np.zeros(self.N+self.m-1)
        pk[:self.N] = self.beta * self.Pvu.T @ (self.Pvs @ state_vector - Vref) + self.gamma * self.Pzu.T @ (self.Pzs @ state_vector - Uc[:,0] * position_fc)
        pk[self.N:] = - self.gamma * U.T @ (self.Pzs @ state_vector - Uc[:, 0] * position_fc)
        
        Qk = np.zeros((self.N+self.m-1, self.N+self.m-1))
        Qk[:self.N, :self.N] = self.alpha * np.identity(self.N) + self.beta * self.Pvu.T @ self.Pvu + self.gamma * self.Pzu.T @ self.Pzu
        Qk[:self.N, self.N:] = - self.gamma * self.Pzu.T @ U
        Qk[self.N:, :self.N] = - self.gamma * U.T @ self.Pzu
        Qk[self.N:, self.N:] = self.gamma * U.T @ U
        
        u = solve_qp(Qk, pk, G=np.zeros((self.N+self.m, self.N+self.m)), h=np.zeros(self.N+self.m), solver=solver)
        return u