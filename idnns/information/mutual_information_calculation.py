'Calculation of the full plug-in distribuation'

import numpy as np
import multiprocessing
from copy import deepcopy
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


def calc_entropy_for_specipic_t(current_ts, px_i):
	"""Calc entropy for specipic t"""
	b2 = np.ascontiguousarray(current_ts).view(
		np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_current_ts = unique_counts / float(sum(unique_counts))
	p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
	H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
	return H2X


def calc_condtion_entropy(px, t_data, unique_inverse_x):
	# Condition entropy of t given x
	H2X_array = np.array(
		Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
		                           for i in range(px.shape[0])))
	H2X = np.sum(H2X_array)
	return H2X

def guassianMatrix(X, sigma):
  # X = X.astype(np.float64)
  G = np.matmul(X, X.T)
  # print(G)
  # print(np.diag(G).reshape((-1, 1)).T)
  # print(np.diag(G))
  # print(np.diag(G).reshape((-1, 1)))
  # print(2 * G - np.diag(G).reshape((-1, 1)).T)
  # K = 2 * G - np.diag(G).reshape((-1, 1)).T
  # print(K)
  # print(2 * G - np.diag(G))
  # print(K - np.diag(G).reshape((-1, 1)))
  # print(K - np.diag(G))
  # print(2 * (G - np.diag(G)))
  # K = K - np.diag(G)
  # print(K)
  # raise
  # G = self.matrix_multiply_numba(X, X.T)
  
  # K = 2 * G - np.diag(G).T # TODO: need check
  # K = np.exp((1/(2 * sigma ** 2)) * (K - np.diag(G)))
  
  # version 2
  K = 2 * G - np.diag(G).reshape((-1, 1)).T
  K = np.exp((1/(2 * sigma ** 2)) * (K - np.diag(G).reshape((-1, 1))))
  # print(K)
  # K = 2 * (G - np.diag(G))
  # K = np.exp((1/(2 * sigma ** 2)) * K)
  # print(K)
  # raise
  return K


def calc_multivariate_information_new(variable1, variable2, sigma1, sigma2, alpha):
  '''
  Rewrite from matlab
  Input shape: 
  variable1: may be output (samples, classes) or (samples, out_chanels, row, col)
  variable2: feature maps (samples, out_chanels, row, col)
  '''
  variable1 = deepcopy(variable1)
  variable2 = deepcopy(variable2)

  if variable1.ndim == 4:
    d1, d2, d3, d4 = variable1.shape
    variable1 = np.reshape(variable1, (d1, d2*d3*d4))
    variable1 = variable1.T # reverse to unify guassianMatrix
    K_x = np.real(guassianMatrix(variable1.T, sigma1)) / variable1.T.shape[0]
    eigenvalue, _ = np.linalg.eig(K_x) 
    L_x = eigenvalue
    lambda_x = np.abs(np.diag(L_x))
    H_x = (1/(1-alpha)) * np.log((np.sum(lambda_x ** alpha)))
  else:
    # dim == 2
    # variable1 = variable1.T # reverse to unify guassianMatrix
    K_x = np.real(guassianMatrix(variable1.T, sigma1)) / variable1.T.shape[0]
    # L_x, _ = np.linalg.eig(K_x) # trả ra một vector
    L_x, _ = np.linalg.eigh(K_x) # dùng eigh vì trả ra vector đúng chiều :(
    # np.diag: nếu nhận 2D array -> output diag vector, nếu nhận vector -> output diagonal matrix
    # lambda_x = np.abs(np.diag(L_x)) # ko cần diag vì đã trả ra vector rồi
    lambda_x = np.abs(L_x)
    H_x = (1/(1-alpha)) * np.log((np.sum(lambda_x ** alpha)))
    
  if variable2.ndim == 4:
    L = variable2.shape[1]
    K_s = [_ for _ in range(L)]
    # H_s = np.zeros((1, L))
    H_s = [_ for _ in range(L)]
    
    for i in range(L):
      source = variable2[:, [i], :, :]
      source = source.reshape((source.shape[0], source.shape[2] * source.shape[3]))
      source = source.T
      K_s[i] = np.real(guassianMatrix(source.T, sigma2)) / source.T.shape[0]
      L_s, _ = np.linalg.eig(K_s[i])
      lambda_s = np.abs(np.diag(L_s))
      H_s[i] = (1 / (1-alpha)) * np.log(np.sum(lambda_s**alpha))
  else:
    source = variable2
    # source = source.T
    K_s = np.real(guassianMatrix(source.T, sigma2)) / source.T.shape[0]
    # L_s, _ = np.linalg.eig(K_s)
    L_s, _ = np.linalg.eigh(K_s)
    # lambda_s = np.abs(np.diag(L_s))
    lambda_s = np.abs(L_s)
    H_s = (1 / (1-alpha)) * np.log(np.sum(lambda_s**alpha))
    
  if variable2.ndim == 4:
    # estimate joint entropy H(s1,s2,s3...)
    K_sall = K_s[0]
    for i in range(1, L):
      K_sall = K_sall * K_s[i] * variable1.T.shape[0]
    L_sall, _ = np.linalg.eig(K_sall)
    lambda_sall = np.abs(np.diag(L_sall))
    H_sall = (1/(1-alpha)) * np.log(np.sum(lambda_sall ** alpha))
    
    
    # estimate joint entropy H(x,s1,s2,s3...)
    K_xsall = K_x * K_sall * variable1.T.shape[0] # TODO: .T update to other places
    L_xsall, _ = np.linalg.eig(K_xsall)
    # lambda_xsall = np.abs(np.diag(L_xsall))
    lambda_xsall = np.abs(L_xsall) # TODO: update to other places
    H_xsall = (1/(1-alpha)) * np.log(np.sum(lambda_xsall ** alpha))
    
    mutual_information = H_x + H_sall - H_xsall

  else:
    # estimate joint entropy H(x,s)
    K_xs = K_x * K_s * variable1.T.shape[0]
    # L_xs, _ = np.linalg.eig(K_xs)
    L_xs, _ = np.linalg.eigh(K_xs)
    lambda_xs = np.abs(L_xs)
    H_xs = (1/(1-alpha)) * np.log(np.sum(lambda_xs ** alpha))
    mutual_information = H_x + H_s - H_xs

  return mutual_information

def calc_multivariate_information(variable1, variable2, sigma1, sigma2, alpha):
  '''
  Rewrite from matlab
  Input shape: 
  variable1: may be output (samples, classes) or (samples, out_chanels, row, col)
  variable2: feature maps (samples, out_chanels, row, col)
  '''
  variable1 = deepcopy(variable1)
  variable2 = deepcopy(variable2)

  if variable1.ndim == 4:
    d1, d2, d3, d4 = variable1.shape
    variable1 = np.reshape(variable1, (d1, d2*d3*d4))
    variable1 = variable1.T # reverse to unify guassianMatrix
    K_x = np.real(guassianMatrix(variable1.T, sigma1)) / variable1.T.shape[0]
    eigenvalue, _ = np.linalg.eig(K_x) 
    L_x = eigenvalue
    lambda_x = np.abs(np.diag(L_x))
    H_x = (1/(1-alpha)) * np.log((np.sum(lambda_x ** alpha)))
  else:
    # dim == 2
    variable1 = variable1.T # reverse to unify guassianMatrix
    K_x = np.real(guassianMatrix(variable1.T, sigma1)) / variable1.T.shape[0]
    L_x, _ = np.linalg.eig(K_x)
    lambda_x = np.abs(np.diag(L_x))
    H_x = (1/(1-alpha)) * np.log((np.sum(lambda_x ** alpha)))
    
  if variable2.ndim == 4:
    L = variable2.shape[1]
    K_s = [_ for _ in range(L)]
    # H_s = np.zeros((1, L))
    H_s = [_ for _ in range(L)]
    
    for i in range(L):
      source = variable2[:, [i], :, :]
      source = source.reshape((source.shape[0], source.shape[2] * source.shape[3]))
      source = source.T
      K_s[i] = np.real(guassianMatrix(source.T, sigma2)) / source.T.shape[0]
      L_s, _ = np.linalg.eig(K_s[i])
      lambda_s = np.abs(np.diag(L_s))
      H_s[i] = (1 / (1-alpha)) * np.log(np.sum(lambda_s**alpha))
  else:
    source = variable2
    source = source.T
    K_s = np.real(guassianMatrix(source.T, sigma2)) / source.T.shape[0]
    L_s, _ = np.linalg.eig(K_s)
    lambda_s = np.abs(np.diag(L_s))
    H_s = (1 / (1-alpha)) * np.log(np.sum(lambda_s**alpha))
    
  if variable2.ndim == 4:
    # estimate joint entropy H(s1,s2,s3...)
    K_sall = K_s[0]
    for i in range(1, L):
      K_sall = K_sall * K_s[i] * variable1.T.shape[0]
    L_sall, _ = np.linalg.eig(K_sall)
    lambda_sall = np.abs(np.diag(L_sall))
    H_sall = (1/(1-alpha)) * np.log(np.sum(lambda_sall ** alpha))
    
    
    # estimate joint entropy H(x,s1,s2,s3...)
    K_xsall = K_x * K_sall * variable1.T.shape[0] # TODO: .T update to other places
    L_xsall, _ = np.linalg.eig(K_xsall)
    # lambda_xsall = np.abs(np.diag(L_xsall))
    lambda_xsall = np.abs(L_xsall) # TODO: update to other places
    H_xsall = (1/(1-alpha)) * np.log(np.sum(lambda_xsall ** alpha))
    
    mutual_information = H_x + H_sall - H_xsall
  else:
    # estimate joint entropy H(x,s)
    K_xs = K_x * K_s * variable1.T.shape[0]
    L_xs, _ = np.linalg.eig(K_xs)
    lambda_xs = np.abs(L_xs)
    H_xs = (1/(1-alpha)) * np.log(np.sum(lambda_xs ** alpha))
    mutual_information = H_x + H_s - H_xs

  return mutual_information


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array):
	"""Calculate the MI based on binning of the data"""
	H2 = -np.sum(ps2 * np.log2(ps2))
	H2X = calc_condtion_entropy(px, data, unique_inverse_x)
	H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
	IY = H2 - H2Y
	IX = H2 - H2X
	return IX, IY


def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
	"""Calculate the p(x|T) and p(y|T)"""
	indexs = unique_inverse == t_index
	p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
	unique_array_internal, unique_counts_internal = \
		np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
	indexes_x = np.where(np.in1d(b1, b[indexs]))
	p_x_ts = np.zeros(len_unique_a)
	p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
	return p_x_ts, p_y_ts
