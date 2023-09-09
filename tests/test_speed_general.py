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

n_times = 1
Np=60
K=20000

## test GA.*(GC*GB)
RA = np.random.random(size=(Np, K))
RC = np.random.random(size=(Np, K))
RB=np.random.random(size=(Np, Np))



def change_order(RA, RB, RC, orderlist):
    A = np.array(RA, order=orderlist[0])
    B = np.array(RB, order=orderlist[1])
    C = np.array(RC, order=orderlist[2])
    print('A -('+ orderlist[0]+'-order)')
    show_details(A)
    print('B -('+ orderlist[1]+'-order)')
    show_details(B)
    print('C -('+ orderlist[2]+'-order)')
    show_details(C)
    return A,B,C

orderlist="CCC"
todo="A*(B@C)+A*(B@C) "  # rx*(Dr@u)
# todo="""
# A*(B@C)+A*(B@C)
# """  # rx*(Dr@u)
A,B,C=change_order(RA, RB, RC, orderlist)
# timecal=timeit.timeit(lambda: A*(B@C)+A*(B@C), number=n_times)/n_times
timecal=timeit.timeit(lambda: exec(todo), number=n_times)/n_times

print(orderlist+'-order, evaluate: '+todo, timecal) 






# def time_evaluation(todo, orderlist):
#     A = np.array(RA, order=orderlist[0])
#     B = np.array(RB, order=orderlist[1])
#     C = np.array(RC, order=orderlist[2])
#     print('A -('+ orderlist[0]+'-order)')
#     show_details(A)
#     print('B -('+ orderlist[1]+'-order)')
#     show_details(B)
#     print('C -('+ orderlist[2]+'-order)')
#     show_details(C)
#     return timeit.timeit(lambda: A*(B@C)+A*(B@C), number=n_times)/n_times

    # timeha=0
    # exec('timeha=timeit.timeit(lambda:'+ todo+', number=n_times)/n_times')
    # return timeha

    # return timeit.timeit('A*(B@C)+A*(B@C)', number=n_times)/n_times

# timecal=time_evaluation(todo, orderlist)
# print(orderlist+'-order, evaluate: '+todo, timecal) 

# # CCF-order
# A = np.array(RA, order='C')
# C = np.array(RC, order='C')
# B = np.array(RB, order='F')
# print('A (C-order)')
# show_details(A)
# print('C (C-order)')
# show_details(C)
# print('B (F-order)')
# show_details(B)
# print('CCF-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')


# # CFC-order
# A = np.array(RA, order='C')
# C = np.array(RC, order='F')
# B = np.array(RB, order='C')
# print('A (C-order)')
# show_details(A)
# print('C (F-order)')
# show_details(C)
# print('B (C-order)')
# show_details(B)
# print('CFC-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')


# # CFF-order
# A = np.array(RA, order='C')
# C = np.array(RC, order='F')
# B = np.array(RB, order='F')
# print('A (C-order)')
# show_details(A)
# print('C (F-order)')
# show_details(C)
# print('B (F-order)')
# show_details(B)
# print('CFF-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')

# # FCF-order  1st slowest
# A = np.array(RA, order='F')
# C = np.array(RC, order='C')
# B = np.array(RB, order='F')
# print('A (F-order)')
# show_details(A)
# print('C (C-order)')
# show_details(C)
# print('B (F-order)')
# show_details(B)
# print('FCF-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')

# # FFF-order  2nd slowest
# A = np.array(RA, order='F')
# C = np.array(RC, order='F')
# B = np.array(RB, order='F')
# print('A (F-order)')
# show_details(A)
# print('C (F-order)')
# show_details(C)
# print('B (F-order)')
# show_details(B)
# print('FFF-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')



# # FCC-order  3rd slowest
# A = np.array(RA, order='F')
# C = np.array(RC, order='C')
# B = np.array(RB, order='C')
# print('A (F-order)')
# show_details(A)
# print('C (C-order)')
# show_details(C)
# print('B (C-order)')
# show_details(B)
# print('FCC-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')

# # FFC-order  slow
# A = np.array(RA, order='F')
# C = np.array(RC, order='F')
# B = np.array(RB, order='C')
# print('A (F-order)')
# show_details(A)
# print('C (F-order)')
# show_details(C)
# print('B (C-order)')
# show_details(B)
# print('FFC-order, matrix (element-wise) mBltiplication: ', timeit.timeit(lambda: A*(C@B)+A*(C@B), nBmber=n_times)/n_times,'\n')