import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import numpy as np
import torch

from dataset.dataset_skeleton import MyDataset, DatasetOnMemory

class Pendulum:

    def __init__(self, init_theta=0.0, init_dtheta=0.0, duration=20, \
                 w=np.array([0, 0, 0.5, 0.5, 1, 9, 0.5, 0.5]), policy_type='learned', noise_var=0.5, \
                    seed = None):
        self.duration = duration
        self.policy_type = policy_type
        
        # Pendulum parameters
        
        self.m = 1.0
        self.l = 0.5
        self.g_hat = 9.81   

        '''
        If you want to get back to the dim(w)=5 version, just need to set
        self.Wind_x = w[0]*2 # [-2,2]
        self.Wind_y = w[1]*2 # [-2,2]
        self.Cd1 = 1 # [0,1]
        self.Cd2 = 0
        self.Cd3 =  1
        self.g = w[2] + 9 # [8,10]
        self.alpha1 = w[3]/2 + 0.5 # [0,1]
        self.alpha2 =  w[4]/2 + 0.5 # [0,1]

        If you want to get back to the dim(w)=8 version, just need to set
        self.Wind_x = w[0]*2 # [-2,2]
        self.Wind_y = w[1]*2 # [-2,2]
        self.Cd1 = w[2]/2 + 0.5 # [0,1]
        self.Cd2 = w[3]/2 + 0.5 # [0,1]
        self.Cd3 = w[4]/2 + 1 # [0.5, 1.5]
        self.g = w[5] + 9# [8,10]
        self.alpha1 = w[6]/2 + 0.5 # [0,1]
        self.alpha2 = w[7]/2 + 0.5 # [0,1]

        If you want to get back to the close to linear version, just need to set
        self.Wind_x = w[0]*2 # [-2,2]
        self.Wind_y = w[1]*2 # [-2,2]
        self.Cd1 = 1
        self.Cd2 = 0.5
        self.Cd3 =  1
        self.g = w[2] + 9 # [8,10]
        self.alpha1 = w[3]/2 + 0.5 # [0,1]
        self.alpha2 =  w[4]/2 + 0.5 # [0,1]

        If you want to get back to the close to real linear version, just need to set
        self.Wind_x = w[0]*2 # [-2,2]
        self.Wind_y = w[1]*2 # [-2,2]
        self.Cd1 = 1 # [0,1]
        self.Cd2 = 0
        self.Cd3 =  1
        self.g = w[2] + 9 # [8,10]
        self.alpha1 = w[3]/2 + 0.5 # [0,1]
        self.alpha2 =  w[4]/2 + 0.5 # [0,1]
        '''
                                    
        self.Wind_x = w[0]*2 # [-2,2]
        self.Wind_y = w[1]*2 # [-2,2]
        self.Cd1 = 1
        self.Cd2 = 0.5
        self.Cd3 =  1
        self.g = w[2] + 9 # [8,10]
        self.alpha1 = w[3]/2 + 0.5 # [0,1]
        self.alpha2 =  w[4]/2 + 0.5 # [0,1]
    
        # States                 
        self.theta = init_theta                   
        self.dtheta = init_dtheta                                                               
        self.state = np.array([self.theta, self.dtheta])
        self.dstate = np.array([0.0, 0.0])
        self.u = 0.0
        
        # Control gain
        self.gain = 1.5
        
        # Noise
        self.u_noise_sigma = 0.1
        self.u_noise = 0.0
        self.a_noise_sigma = noise_var
        self.a_noise = 0.0
        self.seed = seed
        
        # Step
        self.step_size = 1e-2
        self.total_step = 0 
    
        # Fd data
        self.F_d_data = 0.0
        self.F_d_gt = 0.0
    
    # Ground truth unknown dynamics model
    def F_d(self):
        # External wind velocity
        w_x = self.Wind_x
        w_y = self.Wind_y
        v_x = self.l * self.dtheta * np.cos(self.theta)
        v_y = self.l * self.dtheta * np.sin(self.theta)
        R = np.array([w_x - v_x, w_y - v_y])
        F = self.Cd1 * np.linalg.norm(R) * R + self.Cd2 * R
        F *= self.Cd3
        damping = self.alpha1 * self.dtheta + self.alpha2 * self.dtheta * np.abs(self.dtheta)
        return self.l * np.sin(self.theta) * F[1] + self.l * np.cos(self.theta) * F[0] \
               - damping + self.m * self.l * (self.g - self.g_hat) * np.sin(self.theta)
    
    def noise(self):
        if not self.total_step % 10: 
            if self.seed is not None:
                np.random.seed(self.seed + self.total_step)
            self.u_noise = np.random.normal(0, self.u_noise_sigma)
            if self.u_noise > 3 * self.u_noise_sigma:
                self.u_noise = 3 * self.u_noise_sigma
            if self.u_noise < -3 * self.u_noise_sigma:
                self.u_noise = -3 * self.u_noise_sigma
        if self.seed is not None:
            np.random.seed(self.seed + self.total_step*100)
        self.a_noise = np.random.normal(0, self.a_noise_sigma)
        
    def controller(self):
        u_feedback = self.m * self.l**2 * (-2 * self.gain * self.dtheta - self.gain**2 * self.theta)
        u_feedforward = -self.m * self.l * self.g_hat * np.sin(self.theta)
        u_random = self.u_noise
        
        # TODO: add the learned model here
        F_d_learned = 0.
        u_learned = -F_d_learned
        
        if self.policy_type == 'learned':
            self.u = u_feedback + u_feedforward + u_random + u_learned
        elif self.policy_type == 'random':
            self.u = u_feedback + u_feedforward + 40 * u_random
        elif self.policy_type == 'oracle':
            self.u = u_feedback + u_feedforward + u_random - self.F_d()
        else:
            print('WARNING! No such policy type')
        
    def dynamics(self):
        self.dstate[0] = self.dtheta
        self.dstate[1] = self.u/(self.m*self.l**2) + self.g_hat/self.l*np.sin(self.theta) + self.F_d()/(self.m*self.l**2)
        self.dstate[1] += self.a_noise
        
    # ODE solver: (4,5) Runge-Kutta
    def process(self):
        self.noise()
        self.controller()
        self.F_d_data = self.F_d() + self.a_noise
        self.F_d_gt = self.F_d()
        
        prev_state = self.state
        
        self.dynamics()
        s1_dstate = self.dstate
        
        self.state = prev_state + 0.5 * self.step_size * s1_dstate
        self.dynamics()
        s2_dstate = self.dstate
        
        self.state = prev_state + 0.5 * self.step_size * s2_dstate
        self.dynamics()
        s3_dstate = self.dstate
        
        self.state = prev_state + self.step_size * s3_dstate
        self.dynamics()
        s4_dstate = self.dstate
        
        self.state = prev_state + 1.0 / 6 * self.step_size * \
                      (s1_dstate + 2 * s2_dstate + 2 * s3_dstate + s4_dstate)
        
        self.total_step += 1
        
        self.dtheta = self.state[1]
        self.theta = self.state[0]
                
    def simulate(self):
        Theta = []
        Theta = np.append(Theta, self.theta)
        Dtheta = []
        Dtheta = np.append(Dtheta, self.dtheta)
        Fd_data = []
        Fd_gt = []
        Control = []
        
        while True:
            self.process()
            Theta = np.append(Theta, self.theta)
            Dtheta = np.append(Dtheta, self.dtheta)
            Control = np.append(Control, self.u)
            Fd_data = np.append(Fd_data, self.F_d_data)
            Fd_gt = np.append(Fd_gt, self.F_d_gt)
            
            # if not self.total_step % int(1 / self.step_size * 1.0):
            #    print('Simulation time: ' + str(self.total_step*self.step_size))

            if self.step_size*self.total_step >= self.duration:
                break

        return Theta[:-1], Dtheta[:-1], Control, Fd_data, Fd_gt

class PendulumSimulatorDataset(MyDataset):

    dataset_name = "pendulum_simulator"

    def __init__(self, input_aug_kernel, actual_target, batch_size= 100, num_workers=4):
        super(PendulumSimulatorDataset, self).__init__(batch_size, num_workers)
        self.actual_target = actual_target
        self.input_sets = {}
        self.task_dim = None
        self.input_aug_kernel = input_aug_kernel


    def __observe_to_actual(self, observe):
        actual = np.copy(observe)
        if observe[-1] == 1:
            actual = self.actual_target
            print("generate actual: ", actual)
        return actual

    def generate_synthetic_data(self, task_dict, noise_var=None, seed=None):
        """
        Generate synthetic data and stores into datasets.
        :param task_dict: dictionary of task name and the corresponding (w, n). 
            w is the weight vector for the task and n is the number of examples for the task. 
        :param float noise_var: variance of the noise added to the labels.
        """

        seed = np.random.RandomState(seed).randint(1000000000) if seed is not None else None
        if self.task_dim is None:
            self.task_dim = task_dict[next(iter(task_dict))][0].shape[0]

        for task_name in task_dict:
            w, n  = task_dict[task_name]
            pendulum = Pendulum(duration=n*1e-2, w=self.__observe_to_actual(w), policy_type="random", noise_var=noise_var, seed=seed)
            theta, dtheta, _, Fd_data, Fd_gt = pendulum.simulate()
            inputs = self.input_aug_kernel(np.stack([theta, dtheta], axis=1))
            labels = Fd_data 

            # print("w.shape", w.shape) #debug

            # Store the generated data if the corresponding task does not exist.
            # Otherwise concatenate the new data to the existing data.
            # Note that the input indices are also stored for the task instead of the real input data.
            if task_name in self.input_sets:
                self.input_sets[task_name] = np.concatenate((self.input_sets[task_name], inputs), axis=0)
            else:
                self.input_sets[task_name] = inputs

            if task_name in self.label_sets:
                self.label_sets[task_name] = np.concatenate((self.label_sets[task_name], labels), axis=0)
            else:
                self.label_sets[task_name] = labels

            # Also update the sampled tasks dictionary to store all the sampled tasks with name and parameter.
            if "test" in task_name:
                self.sampled_test_tasks.update({task_name: w})
            elif "val" in task_name:
                self.sampled_val_tasks.update({task_name: w})
            else:
                self.sampled_train_tasks.update({task_name: w})

    def get_dataset(self, task_name_list, mixed):
        """
        Get dataset for the task.
        :param str task_name: name of the task
        :param bool mixed: whether to mix the data from different tasks.
        :return: dataset for the tasks.
        """

        if mixed:
            total_inputs = None
            total_labels = None
            for task_name in task_name_list:
                if total_inputs is None:
                    total_inputs = self.input_sets[task_name]
                else:
                    total_inputs = np.concatenate((total_inputs, self.input_sets[task_name]), axis=0)
                if total_labels is None:
                    total_labels = self.label_sets[task_name]
                else:
                    total_labels = np.concatenate((total_labels, self.label_sets[task_name]), axis=0)
            total_ws = np.empty((len(total_labels), self.task_dim))
            counter = 0
            for task_name in task_name_list:
                if "test" in task_name:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_test_tasks[task_name].T
                elif "val" in task_name:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_val_tasks[task_name].T
                else:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_train_tasks[task_name].T
                counter += len(self.label_sets[task_name])
            output = DatasetOnMemory(total_inputs, total_labels, total_ws)
        else:
            output = {}
            for task_name in task_name_list:
                assert (task_name in self.input_sets) and (task_name in self.label_sets), \
                    "Dataset for task {} does not exist. Please generate first".format(task_name)
                total_ws = np.empty((len(self.label_sets[task_name]), self.task_dim))
                if "test" in task_name:
                    total_ws[:,:] = self.sampled_test_tasks[task_name].T
                elif "val" in task_name:
                    total_ws[:,:] = self.sampled_val_tasks[task_name].T
                else:
                    total_ws[:,:] = self.sampled_train_tasks[task_name].T 
                output[task_name] = DatasetOnMemory(self.input_sets[task_name], self.label_sets[task_name], total_ws)
        return output



