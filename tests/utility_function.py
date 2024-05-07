import numpy as np

def show_details(arr: np.ndarray):
    first_col_width = 20
    details = f"{'array':<{first_col_width}s}: {np.array_str(arr).replace(chr(10),',')}\n"\
             +f"{'flags':<{first_col_width}s}: {arr.flags.f_contiguous = }, {arr.flags.c_contiguous = }\n"\
             +f"{'array interface':<{first_col_width}s}: {arr.__array_interface__}\n" \
             +f"{'strides':<{first_col_width}s}: {arr.strides}\n" \
             +f"{'datatype':<{first_col_width}s}: {arr.dtype}\n" \
             +f"{'base':<{first_col_width}s}: {np.array_str(arr.base).replace(chr(10),',') if arr.base is not None else '-'}\n"\
             +f"{'base flags':<{first_col_width}s}: " + (f"{arr.base.flags.f_contiguous = }, {arr.base.flags.c_contiguous = }\n" if arr.base is not None else "-")
    return print( details, sep='\n')
    # print(details, sep='\n')
    # return print(show_details(arr), sep='\n')


# a = np.array([[1,2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# print('-- a', show_details(a), sep='\n')
import timeit
n_times = 10
n_arr = 20000
# rand_array = np.random.random(size=(n_arr, n_arr))

# C-order
# a = np.array(rand_array, order='C')
a = np.ones((n_arr, n_arr), order='C')
print('-- a (C-order)', show_details(a), sep='\n')
# sum over rows
print('C-order, row sum: ', timeit.timeit(lambda: a.sum(axis=1), number=n_times)/n_times)
# sum over columns
print('C-order, column sum: ', timeit.timeit(lambda: a.sum(axis=0), number=n_times)/n_times)

# F-order
# a = np.array(rand_array, order='F')
a = np.ones((n_arr, n_arr), order='F')

print('-- a (F-order)', show_details(a), sep='\n')
# sum over rows
print('F-order, row sum: ', timeit.timeit(lambda: a.sum(axis=1), number=n_times)/n_times)
# sum over columns
print('F-order, column sum: ', timeit.timeit(lambda: a.sum(axis=0), number=n_times)/n_times)