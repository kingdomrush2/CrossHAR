import numpy as np
import torch


def DataTransform(sample, config=None):
    # shape(data_size,channel_num,seq_len)
    sample = sample.transpose((0, 2, 1))
    weak_aug = scaling(sample, sigma=1.1)
    strong_aug = jitter(permutation(sample, max_segments=8), sigma=0.01)
    weak_aug = weak_aug.transpose((0, 2, 1))
    strong_aug = strong_aug.transpose((0, 2, 1))
    return weak_aug.astype(np.float32), strong_aug.astype(np.float32)


def jitter(x, sigma=0.8):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return ret
