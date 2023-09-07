import numpy as np
import timeit

def show_details(arr: np.ndarray):
    first_col_width = 20
    # details = f"{'flags':<{first_col_width}s}: {arr.flags.f_contiguous = }, {arr.flags.c_contiguous = }\n"\
    #         +f"{'strides':<{first_col_width}s}: {arr.strides}\n" \
    details = f"{'flags':<{first_col_width}s}: {arr.flags.f_contiguous = }, {arr.flags.c_contiguous = }\n"\
            #  +f"{'datatype':<{first_col_width}s}: {arr.dtype}\n" \
            #  +f"{'base':<{first_col_width}s}: {np.array_str(arr.base).replace(chr(10),',') if arr.base is not None else '-'}\n"\
            #  +f"{'base flags':<{first_col_width}s}: " + (f"{arr.base.flags.f_contiguous = }, {arr.base.flags.c_contiguous = }\n" if arr.base is not None else "-")
    return print( details, sep='\n')

n_times = 8
Np=60
K=2000000

## test Grx.*(GDr*Gu)
Rrx = np.random.random(size=(Np, K))
Ru = np.random.random(size=(Np, K))
RDr=np.random.random(size=(Np, Np))

# CCC-order
rx = np.array(Rrx, order='C')
Dr = np.array(RDr, order='C')
u = np.array(Ru, order='C')
print('rx (C-order)')
show_details(rx)
print('Dr (C-order)')
show_details(Dr)
print('u (C-order)')
show_details(u)
print('CCC-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')

# CCF-order
rx = np.array(Rrx, order='C')
Dr = np.array(RDr, order='C')
u = np.array(Ru, order='F')
print('rx (C-order)')
show_details(rx)
print('Dr (C-order)')
show_details(Dr)
print('u (F-order)')
show_details(u)
print('CCF-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')


# CFC-order
rx = np.array(Rrx, order='C')
Dr = np.array(RDr, order='F')
u = np.array(Ru, order='C')
print('rx (C-order)')
show_details(rx)
print('Dr (F-order)')
show_details(Dr)
print('u (C-order)')
show_details(u)
print('CFC-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')


# CFF-order
rx = np.array(Rrx, order='C')
Dr = np.array(RDr, order='F')
u = np.array(Ru, order='F')
print('rx (C-order)')
show_details(rx)
print('Dr (F-order)')
show_details(Dr)
print('u (F-order)')
show_details(u)
print('CFF-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')

# # FCF-order  1st slowest
# rx = np.array(Rrx, order='F')
# Dr = np.array(RDr, order='C')
# u = np.array(Ru, order='F')
# print('rx (F-order)')
# show_details(rx)
# print('Dr (C-order)')
# show_details(Dr)
# print('u (F-order)')
# show_details(u)
# print('FCF-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')

# # FFF-order  2nd slowest
# rx = np.array(Rrx, order='F')
# Dr = np.array(RDr, order='F')
# u = np.array(Ru, order='F')
# print('rx (F-order)')
# show_details(rx)
# print('Dr (F-order)')
# show_details(Dr)
# print('u (F-order)')
# show_details(u)
# print('FFF-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')



# # FCC-order  3rd slowest
# rx = np.array(Rrx, order='F')
# Dr = np.array(RDr, order='C')
# u = np.array(Ru, order='C')
# print('rx (F-order)')
# show_details(rx)
# print('Dr (C-order)')
# show_details(Dr)
# print('u (C-order)')
# show_details(u)
# print('FCC-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')

# # FFC-order  slow
# rx = np.array(Rrx, order='F')
# Dr = np.array(RDr, order='F')
# u = np.array(Ru, order='C')
# print('rx (F-order)')
# show_details(rx)
# print('Dr (F-order)')
# show_details(Dr)
# print('u (C-order)')
# show_details(u)
# print('FFC-order, matrix (element-wise) multiplication: ', timeit.timeit(lambda: rx*(Dr@u)+rx*(Dr@u), number=n_times)/n_times,'\n')