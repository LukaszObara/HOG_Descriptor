# HOG.py

# Libraries
# Standard Libraries
from functools import reduce
import os

# Third-Party Libraries
import numpy as np
from scipy import ndimage
from sklearn.externals import joblib

def grad(image):
	"""
	Computes the gradients of an image

	paramaters
	----------
	type image: 2D numpy array
	param image: Grayscale image 

	returns
	----------
	(np.array, np.array)

	returns a 2-tuple of the (horzotal gradient, vertical gradient)
	"""

	grad_hor, grad_ver = np.zeros_like(image), np.zeros_like(image)

	grad_hor = np.gradient(image, axis=1)
	grad_hor[:, 1:-1] = (2 * grad_hor[:, 1:-1]) % 256
	grad_ver = np.gradient(image, axis=0)
	grad_ver[1:-1, :] = (2 * grad_ver[1:-1, :]) % 256

	return grad_hor, grad_ver    

def magnitude_orientation(gx, gy):
	"""
	Computes the magnitude and angle of the gradient

	paramaters
	----------
	type gx: 2D numpy array
	param image: x component of gradient

	type gy: 2D numpy array
	param image: y component of gradient

	returns
	----------
	(np.array, np.array)

	returns a 2-tuple of the (magintude, orientation)
	"""

	magnitude = np.hypot(gx, gy)
	orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180

	return magnitude, orientation

def _build_cell(orientation, gradient, nbins):
	"""
	Computes the vote per cell

	paramaters
	----------
	type orientation: 2D numpy array
	param orientation: orientation matrix

	type gradient: 2D numpy array
	param gradient: gradient matrix 

	type nbins: int
	param nbins: number of bins

	returns
	----------
	np.array

	returns the histogram for the given cell
	"""

	cell_histogram = np.zeros(shape=nbins)
	b_step = 180/nbins

	for orient, grad in zip(orientation, gradient):
		for o, g in zip(orient, grad):
			c_bin = np.int8(np.abs(o//20))
			prop = np.abs(o % b_step) / b_step

			if c_bin == (nbins -1):
				cell_histogram[c_bin] += (1-prop) * g
				cell_histogram[0] += prop * g
			else:
				cell_histogram[c_bin] += (1-prop) * g
				cell_histogram[c_bin+1] += prop * g

	return cell_histogram

def _build_hist(orientation, gradient, nbins, cell_size):
	"""
	Builds the histogram of a given cell

	paramaters
	----------
	type image: 2D numpy array
	param image: image

	type nbins: int
	param nbins: number of bins

	type cell_size: 2D tuple or list
	param cell_size: number of pixels to include in horizontal and 
					 vertical direction

	returns
	----------
	np.array

	returns the computed histograms. 
	"""
	
	cell_y, cell_x = cell_size    
	overfill = tuple([s % c for s, c in zip(orientation.shape, cell_size)])
	
	# Gets the left side of the image, if the image cannot be covered by
	# equal sized cells
	shape = tuple([s_i - over_i 
				   for s_i, over_i in zip(orientation.shape, overfill)])

	histograms = np.zeros((shape[0]//cell_y, shape[1]//cell_x, nbins))
	i = 0
	for y in range(0, shape[0], cell_y):
		j = 0
		for x in range(0, shape[1], cell_x):
			orient = orientation[y: y+cell_y, x: x+cell_x]
			grad = gradient[y: y+cell_y, x:x+cell_x]
			histograms[i, j, :] += _build_cell(orient, grad, nbins=nbins)
			j += 1
		i+= 1

	return histograms

def _build_block(orientation, gradient, nbins, cell_size, block_size):
	"""
	Groups the given cells into appropriate sized blocks

	paramaters
	----------
	type image: 2D numpy array
	param image: image

	type nbins: int
	param nbins: number of bins

	type cell_size: 2D tuple or list
	param cell_size: number of pixels to include in horizontal and 
					 vertical direction

	type block_size: 2D tuple or list
	param block_size: number of cells to include in horizontal and 
					  vertical direction	

	returns
	----------
	np.array

	returns the hog descriptor as a matrix. 
	"""

	histograms = _build_hist(orientation, gradient, nbins, cell_size)
	nbr_y, nbr_x, _ = histograms.shape
	dim = reduce(lambda x, y: x*y, block_size)
	hog_matrix = np.zeros(((nbr_y-1)*(nbr_x-1), dim*nbins))

	i = 0 
	for y in range(0, nbr_y-1):
		j = 0

		for x in range(0, nbr_x-1):
			block = histograms[y: y+block_size[0], x: x+block_size[1]]
			norm_value = np.linalg.norm(block.ravel() + 1e-4, ord=2)
			block = block.ravel()/norm_value
			position = (nbr_x-1)*i + j
			hog_matrix[position] = block
			j += 1			
		i+= 1

	return hog_matrix

def hog(image, nbins=9, cell_size=(8, 8), block_size=(2, 2), flatten=True):
	"""
	Computes the histogram of oriented gradients

	paramaters
	----------
	type image: 2D numpy array
	param image: image

	type nbins: int
	param nbins: number of bins

	type cell_size: 2D tuple or list
	param cell_size: number of pixels to include in horizontal and 
					 vertical direction

	type block_size: 2D tuple or list
	param block_size: number of cells to include in horizontal and 
					  vertical direction	

	type flatten: bool
	param flatten: 

	returns
	----------
	np.array

	returns the hog descriptor as either a matrix or vector. 
	"""

	gx, gy = grad(image)
	mag, orientation = magnitude_orientation(gx, gy)

	hog_matrix = _build_block(orientation=orientation, gradient=mag, 
							  nbins=nbins, cell_size=cell_size, 
							  block_size=block_size)

	if flatten:
		return hog_matrix.ravel()
	else:
		return hog_matrix

def hog_svm(data, seed=35, save=False, save_loc='C:\\Users\\lukas\\OneDrive\\' \
			  + 'Documents\\Machine Learning\\Pedestrian_detection\\Datasets\\' \
			  + 'INRIA\\96X160H96\\Train\\model.pkl'):
	"""
	Computes the the support vector classifier for the transformed 
	images.
	"""

	from sklearn import svm
	from sklearn.model_selection import KFold

	rng = np.random.RandomState(seed=seed)
	random_index = rng.randint(0, data.shape[0], 4930)
	data = data[random_index]

	kf = KFold(n_splits=10)
	clf = svm.SVC(kernel='linear')
	coefficients = np.zeros(shape=(1, 7524))

	for train, test in kf.split(data):
		train_data = data[train]
		test_data = data[test]

		clf.fit(train_data[:, :-1], train_data[:, -1])
		predictions = [int(a) for a in clf.predict(test_data[:, :-1])]
		num_correct = sum(int(a==y) for a, y in zip(predictions, test_data[:, -1]))
		print(num_correct)

	if save:
		from sklearn.externals import joblib

		print("Saving Model")
		joblib.dump(clf, save_loc)

def sliding_window(image, window_size, step_size):
	"""
	Slides a window across the image

	paramaters
	----------
	type image: numpy array
	param image: image 

	type window_size: 2-D tuple or list
	param window_size: x, y size of window

	type step_size: int
	param step_size: stride by which window will be translated in both
					 x and y direction
	
	returns
	---------
	tupe (int, int, np.array)

	Generator that returns the x, y coordinate and the image within the 
	window. 
	"""

	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# yield the current window
				img = image[y: y+window_size[1], x: x+window_size[0]]
				if img.shape == (int(window_size[1]), int(window_size[0])):
					
					yield (x, y, img)

def main():
	loc_p = 'C:\\...\\Train\\pos\\'
	loc_n = 'C:\\...\\Train\\neg\\'
	loc_s = 'C:\\...\\Train\\data.npy'
	loc_c = 'C:\\...\\Train\\coef.npy'

	p_files = os.listdir(loc_p)
	n_files = os.listdir(loc_n) 
	total_files = len(p_files) + len(n_files)
	data = np.zeros(shape=(total_files, 7525))

	print("Starting to process postive training data")
	for i, files in enumerate(p_files):
		print("{}, {}".format(i, files))
		file = loc_p + str(files)
		img = ndimage.imread(file, mode='L')
		hog_descriptor = hog(img)

		data[i, :-1] += hog_descriptor
		data[i, -1] = -1

	print("\nStarting to process negative training data")
	for file in n_files:
		i += 1
		print("{}, {}".format(i, file))
		file = loc_n + str(file)
		img = ndimage.imread(file, mode='L')
		hog_descriptor = hog(img)

		data[i, :-1] += hog_descriptor
		data[i, -1] = 1
		
	np.save(loc_s, data)

	# training SVC
	data = np.load(loc_s)
	hog_svm(data, save=True)
  
	loc_t = 'C:\\...\\Train\\crop_000608.png'
	img = ndimage.imread(loc_t, mode='L')

	save_loc = "C:\\...\\Train\\model.pkl"
	clf = joblib.load(save_loc) 

	# sl_window = sliding_window(img, window_size=(96, 160), step_size=8)
	# for window in sl_window:
	# 	tmp = window[-1]
	# 	hog_im = hog(tmp)
	# 	hog_im = hog_im.reshape(1, -1)
	# 	predict = clf.predict(hog_im)
	# 	print(predict)

if __name__ == "__main__":
	main()
