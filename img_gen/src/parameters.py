import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


def get_stellar_density():

    mags = np.array([0, 1.7652329749103939, 2.1236559139784945, 2.6433691756272406, 3.3781362007168454, 4.578853046594982, 5.743727598566308, 7.195340501792114, 8.503584229390682, 9.883512544802866, 11.17383512544803, 12.5,              13.79032258064516, 15.474910394265233, 16.56810035842294, 17.67921146953405, 18.915770609318994, 20.026881720430108, 20.815412186379927, 21.335125448028673, 21.747311827956985, 22.051971326164875, 22.374551971326163, 22.625448028673834, 22.91218637992831])

    nstars = np.array([0, 2.0444674699712206 , 6.339394541558143 , 17.334361519909546 , 41.798472357705144 ,156.50897570382014 ,586.0276247953254 ,3004.80089302922 ,11251.088459342604 ,44861.84526680207 ,139105.63077464644 ,431332.60342127783 ,1142932.2249602748 , 3773912.345819709 ,7776514.443049915 ,15528384.633640364 , 34074006.992518626 , 61916852.86229318 , 72454983.08284084 , 2211616.8882064833 , 590651.1556402461 , 37150.603258297764 , 1326.9876903441066 , 73.60277658786532 ,2.0444674699712206])

    total_area = 4*np.pi

    star_density_mags = nstars / total_area

    mag_bins = np.arange(0, 21, 1)
    print(mag_bins)

    mags_interp = interp1d(mags, star_density_mags, kind='linear')

    total_star_mags = []
    for m in mag_bins:
        res, err = quad(mags_interp, 0, m)

        total_star_mags.append(res)


    return mag_bins, total_star_mags

def sample_star_mags(mag_bins, total_star_mags, area, mag_limit = 7):

    nstars = []
    
    for m, n in zip(mag_bins, total_star_mags):
        n *= area
        if m > mag_limit:
            break
        if n < 1:
            nstars.extend([int(m) for ii in sample_subunity(n)])
        else:
            nstars.extend([int(m) for ii in range(int(np.random.normal(loc=n, scale=np.sqrt(n))))])

    return nstars

def sample_subunity(val):

    arr = []
        
    while np.random.random() < val:
        arr.append(0)

    return arr






if __name__ == "__main__":
    mag_bins, total_star_mags = get_stellar_density()
    sample_star_mags(mag_bins, total_star_mags, 1.0, mag_limit=5)
