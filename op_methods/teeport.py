from __future__ import print_function, absolute_import
from mint.mint import *
import nest_asyncio
nest_asyncio.apply()
from teeport import Teeport as Tee

class Teeport(Minimizer):
    def __init__(self, uri=None, optimizer_id=None):
        super(Teeport, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None
        self.uri = uri
        self.opt_id = optimizer_id
    
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
        nvar = len(x)
        g_vrange = np.zeros((nvar, 2))
        for idev, dev in enumerate(self.devices):
            low_limit, high_limit = dev.get_limits()
            if np.abs(low_limit) < 1e-7 and np.abs(high_limit) < 1e-7:
                low_limit, high_limit = -10, 10
            g_vrange[idev, 0], g_vrange[idev, 1] = low_limit, high_limit

        p0 = np.array(x)
        x0 = ((p0 - g_vrange[:, 0])/(g_vrange[:, 1] - g_vrange[:, 0])).reshape(1, -1)

        teeport = Tee(self.uri)  # init a Teeport adapter
        # configs = {
        #     'xtol': self.xtol,
        #     'maxiter': self.max_iter,
        #     'isim': isim,
        #     'x0': x
        # }
        # optimize = teeport.use_optimizer(self.opt_id, configs)
        optimize = teeport.use_optimizer(self.opt_id)

        def evaluate(X, configs=None):
            X = (g_vrange[:, 1] - g_vrange[:, 0]) * X + g_vrange[:, 0]  # denormalize
            Y = []
            for x in X:
                Y.append(-error_func(x))
            Y = np.array(Y).reshape(-1, 1)
            return Y

        optimize(evaluate)
        # teeport.reset()  # clean up

        return x
