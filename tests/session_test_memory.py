# coding: utf-8
import tests.utility_function
show_details(a)
tests.utility_function.show_details(bcv)
import numpy as np
x=np.random.rand(2,4)
x
tests.utility_function.show_details(x)
xt=x.T
xt
tests.utility_function.show_details(xT)
tests.utility_function.show_details(xt)
xt.base
xt.base is x
np.shares_memory(xt.base,x)
np.random.rand(4,2).strides
xvc=x.reshape(-1)
xvc
xv
xvc.base
tests.utility_function.show_details(xvc)
xvf=x.reshape(-1,order='F')
tests.utility_function.show_details(xvf)
xvf.base
np.shares_memory(xvf.base,x)
np.shares_memory(xvc.base,x)
xvf.base is x
xvc.base is x
id(xvc.base)
id(x)
id(xvf.base)
id(xt)
id(xt.base)
xvf2=xt.reshape(-1)
xvf2.base is x
xvf2.base is xt
np.shares_memory(xvf2.base,xt)
tests.utility_function.show_details(xvf2)
xt 
tests.utility_function.show_details(xt)
xvf2.base is xt
xvf2.base 
xvc=x.reshape(-1)
xvc.base is x
xvf3=xt.reshape(-1,order='F')
xvf
xvf3.base is xt
tests.utility_function.show_details(xvf3)
xvf3=xt.reshape(-1,order='C')
tests.utility_function.show_details(xvf3)
xvf3.base is xt
xvf3.base is x
xvf3.base
tests.utility_function.show_details(xt)
tests.utility_function.show_details(x)
tests.utility_function.show_details(xvf)
tests.utility_function.show_details(xvf2)
tests.utility_function.show_details(xvf3)
xvf3 
xvf2 
xvf 
xv
xv=x.reshape(-1)
xv.base is x
xt.base is x
xv
xvc
np.shares_memory(x,xt)
np.shares_memory(x,xvf)
np.shares_memory(x,xvf2)
np.shares_memory(x,xvf3)
np.shares_memory(x,xvf)
np.shares_memory(xt,xvf2)
np.shares_memory(x,xv)
get_ipython().system('cd tests')
get_ipython().run_line_magic('cd', 'tests')
