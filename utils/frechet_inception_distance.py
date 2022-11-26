import os
import numpy as np
import inception_v3
import skimage.transform as transform
import scipy.linalg as linalg
import torch

# scale an array of images to a new size
def scale_images(images, new_shape):
    imgs_shape = (images.shape[0], *new_shape)
    images_list = np.zeros(shape=imgs_shape, dtype=np.float32)
    for i, image in enumerate(images):
        images_list[i] = transform.resize(image, new_shape, preserve_range=True)
    return images_list
#model predict

def model_predict(model, image):
    model.eval()
    result = model(torch.FloatTensor(image).cuda())
    result = result.cpu().detach().numpy()
    return result

# calculate frechet inception distance
def calculate_frechet_inception_distance(model, images1, images2):
	# calculate activations
	act1 = model_predict(model,images1)
	act2 = model_predict(model,images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


class FrechetInceptionDistance:
    def __init__(self):
        self.model = inception_v3.load_and_build_inception_net(pooling='avg')

    def __call__(self, images1, images2):
        # resize images and convert integer to floating point values
        images1 = scale_images(images1, (3,299,299)) #in theano is CxHxW
        images2 = scale_images(images2, (3,299,299)) #in theano is CxHxW
        return calculate_frechet_inception_distance(self.model,images1,images2)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset.datasets import cifar10
    db = cifar10()
    samples = 256
    X = db["X_train"]
    T = db["X_test"]
    print("compute Frechet Inception Distance")
    print(FrechetInceptionDistance()(X[:samples,:,:,:], T[:samples,:,:,:]))