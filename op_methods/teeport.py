from __future__ import print_function, absolute_import
from mint.mint import *
from teeport import Teeport as Tee

class Teeport(Minimizer):
    def __init__(self, uri=None):
        super(Teeport, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None
        self.uri = uri
    
    def preprocess(self):
        """
        defining attribute self.dev_steps

        :return:
        """
        if self.dev_steps is not None:
            return 
        self.dev_steps = []
        for dev in self.devices:
            if "istep" not in dev.__dict__:
                self.dev_steps = None
                return
            elif dev.istep is None or dev.istep == 0:
                self.dev_steps = None
                return
            else:
                self.dev_steps.append(dev.istep)
    
    def minimize(self,  error_func, x):
        #print("start seed", np.count_nonzero(self.dev_steps))
        if self.dev_steps == None or len(self.dev_steps) != len(x):
            print("initial Teeport is None")
            isim = None
        elif np.count_nonzero(self.dev_steps) != len(x):
            print("There is zero step. Initial Teeport is None")
            isim = None
        else:
            #step = np.ones(len(x))*0.05
            isim = np.zeros((len(x) + 1, len(x)))
            isim[0, :] = x
            for i in range(len(x)):
                vertex = np.zeros(len(x))
                vertex[i] = self.dev_steps[i]
                isim[i + 1, :] = x + vertex
            print("ISIM = ", isim)

        nvar = len(x)
        g_vrange = np.zeros((nvar, 2))
        for idev, dev in enumerate(self.devices):
            low_limit, high_limit = dev.get_limits()
            if np.abs(low_limit) < 1e-7 and np.abs(high_limit) < 1e-7:
                low_limit, high_limit = -10, 10
            g_vrange[idev, 0], g_vrange[idev, 1] = low_limit, high_limit
        p0 = np.array(x)
        x0 = ((p0 - g_vrange[:, 0])/(g_vrange[:, 1] - g_vrange[:, 0])).reshape(1, -1)

        uri = self.uri or 'ws://localhost:8080/'
        teeport = Tee(uri)  # init a Teeport adapter
        optimizer_id = ''
        configs = {
            'xtol': self.xtol,
            'maxiter': self.max_iter,
            'isim': isim,
            'x0': x
        }
        optimize = teeport.use_optimizer(optimizer_id, configs)
        res = optimize(error_func)
        teeport.reset()  # clean up

        return res
