import numpy as np

ueV_to_MHz = 1e3/4

def get_spectrum(signal, time_step , total_time):
    '''
    This function calculates the spectrum of a signal.
    ----------------
    Parameters:
    signal: the signal
    time_step: time step
    total_time: total time of the signal
    ----------------
    Returns:
    f: the frequencies
    Sxx: the spectrum
    '''
    N = len(signal)
    f = np.fft.fftfreq(N, time_step)
    xf = np.fft.fft(signal)
    Sxx = 2*time_step**2/total_time*(xf*np.conj(xf))
    Sxx = Sxx.real
    Sxx = Sxx[:int(N/2)]
    return f[:int(N/2)], Sxx

class Telegraph_Noise():
    def __init__(self, sigma, gamma, x0=None):
        self.gamma = gamma
        self.x0 = x0
        self.sigma = sigma
        if x0 is None:
            self.x = self.sigma*(2*np.random.randint(0, 2)-1)

    def update(self, dt):
        # update telegraph noise
        switch_probability = 1/2 - 1/2*np.exp(-2*self.gamma*dt)
        r = np.random.rand() 
        if r < switch_probability:
            self.x = -self.x    
        return self.x
    
    def reset(self):
        if self.x0 is None:
            self.x = self.sigma*(2*np.random.randint(0, 2)-1)
        else:
            self.x = self.x0




class OU_noise():
    def __init__(self, sigma, gamma, x0=None):
        self.tc = 1/gamma
        self.sigma = sigma
        self.x0 = x0
        if x0 is None:
            self.x = np.random.normal(0,sigma)
        else:
            self.x = x0

        self.name = f'ou_tc_{self.tc}_sigma_{self.sigma}'
        self.constructor = np.array([sigma, gamma])

  
    def update(self, dt):
        self.x = self.x*np.exp(-dt/self.tc) + np.sqrt(1-np.exp(-2*dt/self.tc))*np.random.normal(0,self.sigma)
        return self.x

    def reset(self, x0=None):
        if x0 is None:
            self.x = np.random.normal(0,self.sigma)
        else:
            self.x = x0
        return self.x

    def set_x(self, x0):
        self.x = x0
        return self.x

    def update_mu(self, dt, mu, std):
        return mu*np.exp(-dt/self.tc)
    
    def update_std(self, dt, mu, std):
        return np.sqrt(self.sigma**2 + (std**2 - self.sigma**2)*np.exp(-2*dt/self.tc))
    
    def update_and_integrate(self, t):
        
        if t/self.tc > 10:
            avg = self.x*self.tc*(1-np.exp(-t/self.tc))
            sig = np.sqrt(2*self.sigma**2/self.tc)
            mu = 1/self.tc
            std2 = sig**2/2/mu**3*(2*mu*t - 3 + 4*np.exp(-mu*t) - np.exp(-2*mu*t))
            self.update(t)
            random = np.random.normal(loc = avg, scale=np.sqrt(std2))
            return random
        elif self.tc/t > 10:
            x_old = self.x
            self.update(t)
            return (x_old + self.x)*t/2
        else:
            x_old = self.x
            dt = self.tc / 100
            times = np.arange(dt, t + dt, dt)
            N = len(times)
            wiener_process = np.sqrt(2 * self.sigma**2 * dt / self.tc) * np.random.normal(0, 1, size=N)
            exp_factors = np.exp((times - t) / self.tc)
            self.x = self.x * np.exp(-t / self.tc) + np.dot(wiener_process, exp_factors)
            res = (x_old - self.x + np.sum(wiener_process)) * self.tc
            return res





  
        

class Over_f_noise():

    def __init__(self, n_fluctuators, S1 ,sigma_couplings, ommax, ommin,
                    fluctuator_class = OU_noise, x0=None, equally_dist = False):
        self.n_telegraphs = int(n_fluctuators)
        self.S1 = S1 * ueV_to_MHz
        self.sigma = np.sqrt(2*S1*np.log(ommax/ommin))
        self.sigma_couplings = sigma_couplings
        self.ommax = ommax
        self.ommin = ommin
        self.equally_dist = equally_dist
        self.fluctuator_class = fluctuator_class
        self.spawn_fluctuators(int(n_fluctuators), sigma_couplings)
        self.cs = [0,0,0]
        if x0 is None:
            self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        else:
            self.x = x0
        self.sigma = np.sqrt(np.sum([fluctuator.sigma**2 for fluctuator in self.fluctuators]))
        self.name = f'over_f_n_{n_fluctuators}_S1_{S1}_sigma_couplings_{sigma_couplings}_ommax_{ommax}_ommin_{ommin}'
        self.constructor = np.array([int(n_fluctuators), S1, sigma_couplings, ommax, ommin])
        
    def spawn_fluctuators(self, n_fluctuator, sigma_couplings):
        uni = np.random.uniform(0,1,size = n_fluctuator)
        if self.equally_dist:
            gammas = self.ommin*np.exp(np.log(self.ommax/self.ommin)*np.linspace(0,1,n_fluctuator))
        else:
            gammas = self.ommax*np.exp(-np.log(self.ommax/self.ommin)*uni)
        sigmas = self.sigma/np.sqrt(n_fluctuator)*np.random.normal(1,sigma_couplings, size=n_fluctuator)
        self.fluctuators = []
        for n, gamma in enumerate(gammas):
            self.fluctuators.append(self.fluctuator_class(sigmas[n], gamma))
        
    def update(self, dt):
        self.x = np.sum([fluctuator.update(dt) for fluctuator in self.fluctuators])
        return self.x
    
    def reset(self, x0=None):
        for fluctuator in self.fluctuators:
            fluctuator.reset(x0)
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return self.x

    def update_mu(self, dt, mu, std):
        return mu
    
    def update_std(self, dt, mu, std):
        std = self.sigma
        return std
        
    def set_x(self, x0):
        x0s = x0*np.random.normal(1, 0.5, size = len(self.fluctuators))/len(self.fluctuators)
        for n,fluctuator in enumerate(self.fluctuators):
            fluctuator.set_x(x0s[n])
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return self.x

      
    def gen_trajectory(self, times):
        trajectory = []
        for time in times:
            trajectory.append(self.update(time))
        return trajectory
    
    def update_and_integrate(self, t):
        I = 0
        for fluctuator in self.fluctuators:
            I += fluctuator.update_and_integrate(t)
        self.x = np.sum([fluctuator.x for fluctuator in self.fluctuators])
        return I
    
    def ideal_spectrum(self, oms):
        Sxx = np.zeros(len(oms))
        for fluctuator in self.fluctuators:
            Sxx += fluctuator.sigma**2*fluctuator.tc/(fluctuator.tc**2*oms**2+1)
        return Sxx

 