import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import numpy as np
import cv2







class SkyImager:
    def __init__(self, sensor_cfg = {}, optics_cfg = {}, exposure_cfg = {}):
        self.sensor_cfg = sensor_cfg
        self.optics_cfg = optics_cfg
        self.exposure_cfg = exposure_cfg

        self.fill_defaults()

    def fill_defaults(self):
        sensor_cfg_default = {
            "nx":720,
            "ny":1024,
            "pix_size":5.4, #in micron
            "bias": 1,
            "dark": 0,
            "read": 0,
        }

        optics_cfg_default = {
            "focal_length":12, # in mm
            "aperture": 20, # in mm
        }

        exposure_cfg_default = {
            "gain": 100,
            "exp_time": 1, # in sec
        }

        for key in sensor_cfg_default:
            if key not in self.sensor_cfg:
                self.sensor_cfg[key] = sensor_cfg_default[key]
        for key in optics_cfg_default:
            if key not in self.optics_cfg:
                self.optics_cfg[key] = optics_cfg_default[key]
        for key in exposure_cfg_default:
            if key not in self.exposure_cfg:
                self.exposure_cfg[key] = exposure_cfg_default[key]

    def __repr__(self):

        result = "".join([f"{key}:  {cfg[key]}\n" 
                    for cfg in [self.sensor_cfg, self.optics_cfg, self.exposure_cfg] 
                        for key in cfg])
        return result.strip()


    def generate_image(self):
        image = np.zeros((self.sensor_cfg["nx"], self.sensor_cfg["ny"]))

        image += self.sensor_cfg["bias"]


        for ii in range(40):
            x = np.random.randint(720)
            y = np.random.randint(1024)
            image = add_star(image, (x, y))
        for ii in range(1000):
            x = np.random.randint(720)
            y = np.random.randint(1024)
            image = add_star(image, (x, y), flux_scale=0.2)



        # add noise
        image += (np.sqrt(image)*np.random.random(size=np.shape(image)))


        return image


def psf_convolve(image):

    image_conv = cv2.GaussianBlur(image, (65, 65), 1, 1)
    return image_conv


def add_star(image, loc, flux_scale=1):
    cutout_size = 65
    flux = flux_scale*50*np.random.pareto(1)
    star_cutout = np.zeros((cutout_size, cutout_size))

    star_cutout[cutout_size //2, cutout_size //2] = flux

    star_cutout = psf_convolve(star_cutout)



    star_min_xedge = loc[0] - cutout_size // 2
    star_max_xedge = loc[0] + cutout_size // 2+1
    star_min_yedge = loc[1] - cutout_size // 2
    star_max_yedge = loc[1] + cutout_size // 2+1

    cutout_min_xedge = 0
    cutout_max_xedge = cutout_size 
    cutout_min_yedge = 0
    cutout_max_yedge = cutout_size 

    img_shape = np.shape(image)

    if star_min_xedge < 0:
        cutout_min_xedge -= star_min_xedge
        star_min_xedge = 0

    if star_max_xedge > img_shape[0]:
        cutout_max_xedge -= (star_max_xedge - img_shape[0])
        star_max_xedge = img_shape[0]

    if star_min_yedge < 0:
        cutout_min_yedge -= star_min_yedge
        star_min_yedge = 0

    if star_max_yedge > img_shape[1]:
        cutout_max_yedge -= (star_max_yedge - img_shape[1])
        star_max_yedge = img_shape[1]

    image[star_min_xedge:star_max_xedge, star_min_yedge:star_max_yedge] += (
        star_cutout[cutout_min_xedge:cutout_max_xedge, cutout_min_yedge:cutout_max_yedge])
    return image


if __name__ == "__main__":
    ski = SkyImager()

    img = ski.generate_image()
    fig, ax = plt.subplots()
    scaler = ZScaleInterval()
    limits = scaler.get_limits(img)
    ax.imshow(img, origin='lower', vmin=limits[0], vmax=1.2*limits[1], cmap="Greys_r")
    plt.show()



