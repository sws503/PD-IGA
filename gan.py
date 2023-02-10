import torch
import os
import random
from torchvision import transforms
from PIL import Image

class VideoSeq:
    def __init__(self, test_data):
        self.dataset = test_data

    def test(self, individual):
        merged_image = Image.new('RGB', (10 * 256, 256), (250, 250, 250))

        for idx, order in enumerate(individual):
            img_file = self.dataset[order]['image']
            img = Image.open(img_file)
            img = img.resize((256, 256))
            img_size = img.size
            merged_image.paste(img,(idx * 256, 0))

        return merged_image

    def mergeimg_init(self, individual):
        img_batch = []

        for idx, order in enumerate(individual):
            merged_image = Image.new('RGB', (256, 256), (250, 250, 250))
            img_file = self.dataset[order]['image']
            img = Image.open(img_file)
            img = img.resize((256, 256))
            img_size = img.size
            merged_image.paste(img,(0, 0))
            img_batch.append(merged_image)

        return img_batch

def get_dataset_archieve(data = 'psy'):

    dataset_path = "C:/Users/WooseokPC/Desktop/project/interactive-evolution/dataset/"

    if data == 'pororo':
        Es = ["Pororo_ENGLISH1_1", "Pororo_ENGLISH1_2", "Pororo_ENGLISH1_3", "Pororo_ENGLISH2_1", "Pororo_ENGLISH2_2",
              "Pororo_ENGLISH2_3", "Pororo_ENGLISH2_4", "Pororo_ENGLISH3_1", "Pororo_ENGLISH3_2", "Pororo_ENGLISH3_3",
              "Pororo_ENGLISH3_4", "Pororo_ENGLISH4_1", "Pororo_ENGLISH4_2", "Pororo_ENGLISH4_3", "Pororo_ENGLISH4_4",
              "Pororo_Rescue", "Pororo_The_Racing_Adventure"]
        for E in Es:

            path_E = dataset_path + "img_pororo/" + E
            eps = os.listdir(path_E)
            #print(eps)

            for ep in eps:
                path_E_ep = os.path.join(path_E, ep)
                gif_files = os.listdir(path_E_ep)
    else:
        seed = 7
        random.seed(seed)
        test_data = []
        test_path = dataset_path + data + "/"
        testing_images = [test_path + f for f in os.listdir(test_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]

    testing_images.sort()
    # print(len(testing_images))

    generated_path = dataset_path + data #+ "_cluster2/"
    img_clustering_dict = {}
    for (path, dir, files) in os.walk(generated_path):
        for filename in files:
            desired_dir_or_file = path[path.rindex('/', 0, -1) + 1:-1] if path.endswith('/') else path[
                                                                                                  path.rindex('/') + 1:]
            img_clustering_dict[desired_dir_or_file] = path + '/' + filename

    for idx, image in enumerate(img_clustering_dict.values()):
        #print(idx, image)
        annotation_data = {'image': image}
        test_data.append(annotation_data)

    return test_data

def get_dataset(data = 'rand_pororo'):

    dataset_path = "C:/Users/WooseokPC/Desktop/project/interactive-evolution/dataset/"
    random.seed()
    test_data = []
    test_path = dataset_path + data + "/"
    testing_images = [test_path + f for f in os.listdir(test_path) if
                      f.endswith(('.jpg', '.jpeg', '.png'))]

    testing_images.sort()

    for idx, image in enumerate(testing_images):
        #print(idx, image)
        annotation_data = {'image': image}
        test_data.append(annotation_data)

    return test_data

def get_initial_model():
    test_data = get_dataset()
    model = VideoSeq(test_data)

    return model

def get_model(model='PGAN', dataset='celebAHQ-512', use_gpu=True):
    """Returns a pretrained GAN from (https://github.com/facebookresearch/pytorch_GAN_zoo).

    Args:
        model (str): Available values are "PGAN", "DCGAN".
        dataset (str: Available values are "celebAHQ-256", "celebAHQ-512', "DTD", "celeba".
            Ignored if model="DCGAN".
        use_gpu (bool): Whether to use gpu.
    """
    test_data = get_dataset()

    #print(test_data)

    #test_data = list(img_clustering_dict.values())


    model = VideoSeq(test_data)
    #model.test([[1,2,3,4,5]])


    return model

def gan_get_model(model='PGAN', dataset='celebAHQ-512', use_gpu=True):
    """Returns a pretrained GAN from (https://github.com/facebookresearch/pytorch_GAN_zoo).

    Args:
        model (str): Available values are "PGAN", "DCGAN".
        dataset (str: Available values are "celebAHQ-256", "celebAHQ-512', "DTD", "celeba".
            Ignored if model="DCGAN".
        use_gpu (bool): Whether to use gpu.
    """
    all_models = ['PGAN', 'DCGAN']
    if not model in all_models:
        raise KeyError(
            f"'model' should be in {all_models}."
        )

    pgan_datasets = ['celebAHQ-256', 'celebAHQ-512', 'DTD', 'celeba']
    if model == 'PGAN' and not dataset in pgan_datasets:
        raise KeyError(
            f"If model == 'PGAN', dataset should be in {pgan_datasets}"
        )

    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', model,
                           model_name=dataset, pretrained=True, useGPU=use_gpu)

    return model

from imagecluster import calc, io as icio, postproc

def test():
    data = 'rand_pororo'
    dataset_path = "C:/Users/WooseokPC/Desktop/project/interactive-evolution/dataset/"
    dataset_path2 = dataset_path + data + "/"
    generated_path = dataset_path + data + "_cluster2/"

    images,fingerprints,timestamps = icio.get_image_data(dataset_path2)
    clusters = calc.cluster(fingerprints, sim=0.5, min_csize=1)
    postproc.make_links(clusters, generated_path)
    postproc.visualize(clusters, images)

def test2():
    # Create image database in memory. This helps to feed images to the NN model
    # quickly.
    data = 'rand_pororo'
    dataset_path = "C:/Users/WooseokPC/Desktop/project/interactive-evolution/dataset/"
    dataset_path2 = dataset_path + data + "/"
    generated_path = dataset_path + data + "_cluster/"

    images = icio.read_images(dataset_path2, size=(224, 224))

    # Create Keras NN model.
    model = calc.get_model()

    # Feed images through the model and extract fingerprints (feature vectors).
    fingerprints = calc.fingerprints(images, model)

    # Optionally run a PCA on the fingerprints to compress the dimensions. Use a
    # cumulative explained variance ratio of 0.95.
    fingerprints = calc.pca(fingerprints, n_components=0.95)

    # Read image timestamps. Need that to calculate the time distance, can be used
    # in clustering.
    timestamps = icio.read_timestamps(generated_path)

    # Run clustering on the fingerprints. Select clusters with similarity index
    # sim=0.5. Mix 80% content distance with 20% timestamp distance (alpha=0.2).
    clusters = calc.cluster(fingerprints, sim=0.5, timestamps=timestamps, alpha=0.2)

    # Create dirs with links to images. Dirs represent the clusters the images
    # belong to.
    postproc.make_links(clusters, generated_path)

    # Plot images arranged in clusters and save plot.
    fig, ax = postproc.plot_clusters(clusters, images)
    fig.savefig('foo.png')
    postproc.plt.show()

def read():

    data = 'rand_pororo'
    dataset_path = "C:/Users/WooseokPC/Desktop/project/interactive-evolution/dataset/"
    dataset_path2 = dataset_path + data + "/"
    generated_path = dataset_path + data + "_cluster/"

    dict = {}
    for (path, dir, files) in os.walk(generated_path):
        for filename in files:
            #print(os.path.basename(path))
            ext = os.path.splitext(filename)[-1]

            desired_dir_or_file = path[path.rindex('/', 0, -1) + 1:-1] if path.endswith('/') else path[path.rindex('/') + 1:]
            dict[desired_dir_or_file] = filename

            #if ext == '.py':
            #print("%s/%s" % (path, filename))
    print(dict)

if __name__ == "__main__":
    get_model()
    #test()