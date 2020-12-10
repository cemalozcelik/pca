import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2 as cv
from sklearn.metrics import mean_squared_error

img=cv.imread("brain.jpg",0)
n=30
pca = PCA(n).fit(img.data)
components = pca.transform(img.data)

projected = pca.inverse_transform(components)

fig, ax = plt.subplots( 1,2, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
print(components)
ax[0].imshow(img.data,cmap="gray")
ax[1].imshow(projected,cmap="gray")
ax[ 0].set_ylabel('full-dim\ninput')
ax[1].set_ylabel(str(n)+"-dim\nreconstruction")
ax[1].set_xlabel("MSE:"+str(mean_squared_error(img,projected)))

plt.show()


