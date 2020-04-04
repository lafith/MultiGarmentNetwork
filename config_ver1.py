# default NUM=8
NUM = 1  # 8  # number of images to extrapolate from
IMG_SIZE = 720
FACE = False

config = lambda: None
config.expName = None
config.checkpoint_dir = None
config.train = lambda: None
config.train.batch_size = 1
config.train.lr = 0.001
config.train.decay = 0.001
config.train.epochs = 10
config.latent_code_garms_sz = 1024

config.PCA_ = 35
config.garmentKeys = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat', 'LongCoat']
config.NVERTS = 27554
