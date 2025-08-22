import sys
import corner
# import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from jax.lax import broadcast_shapes
from jax.scipy.special import ndtr, ndtri
from jax.scipy.stats import norm as jax_norm
from numpyro import deterministic, plate, sample
from numpyro.distributions import (Categorical, Dirichlet, Distribution,
                                   InverseGamma, Normal, constraints,
                                   TruncatedNormal, Uniform)
from numpyro.distributions.util import (is_prng_key, promote_shapes,
                                        validate_sample)
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

from argparse import ArgumentParser
from copy import deepcopy
from os.path import exists, join

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from h5py import File
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from scipy.stats import triang, truncnorm

from jax import numpy as jnp
from jax.scipy.special import erf, logsumexp
from numpyro import deterministic, factor, sample
from numpyro.distributions import Dirichlet

from jax import config
config.update("jax_enable_x64", True)

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

num_bins = int(sys.argv[1])

# gen = np.random.default_rng(42)
# rng_key = random.PRNGKey(42)

gen = np.random.default_rng(np.random.randint(0, 100000000))
rng_key = random.PRNGKey(np.random.randint(0, 100000000))

nwarm, nsamp = 500, 2500

# arr = np.load('stellar_ages.npy')
# age, age_err = arr[:, 0], arr[:, 1]

# arr = np.load('Ages.npy')
# Age1, Age1_err, Age2, Age2_err = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
# # age, age_err = Age1, Age1_err
# age, age_err = Age2, Age2_err

arr = np.load('Ages_DR8.npy')
age_rgb, err_age_rgb, age_rc, err_age_rc = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
age, age_err = age_rgb, err_age_rgb
# age, age_err = age_rc, err_age_rc
mask = (age<20)&(age>0)

print("Initial number of galaxies:", len(age))

age, age_err = age[mask], age_err[mask]
mask = age_err > 0.001
# mask = age_err > 0.5
# mask = (age_err > 0.01)&(age_err < 3)
# mask = age_err > 0.1
age, age_err = age[mask], age_err[mask]

print(np.min(age), np.median(age), np.mean(age), np.max(age))
print(np.min(age_err), np.median(age_err), np.mean(age_err), np.max(age_err))
# quit()

print("Final number of galaxies:", len(age))

xobs, xerr = age, age_err


###############################################################################
#                          Delta function model                               #
###############################################################################

def gauss_intg(a, b, mu, sigma):
    return 0.5 * (
        + erf((b - mu) / (jnp.sqrt(2) * sigma))
        - erf((a - mu) / (jnp.sqrt(2) * sigma))
        )

class DeltaFunctionModel:

    def __init__(self, xobs, xerr, num_bins, low_xobs, high_xobs,
                 bin_edges=None):
        self.xobs = xobs
        self.xerr = xerr

        if bin_edges is not None:
            if xobs.min() < bin_edges[0] or xobs.max() > bin_edges[-1]:
                raise ValueError(
                    "Some values are outside of the binning range.")
        else:
            xmin, xmax = 0, xobs.max()
            print(f"Using {num_bins} bins from {xmin} to {xmax}.", flush=True)
            bin_edges = jnp.linspace(xmin, xmax, num_bins + 1)

        self.bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
        self.bin_edges = bin_edges
        self.bin_widths = bin_edges[1:] - bin_edges[:-1]
        self.log_bin_widths = jnp.log(self.bin_widths)
        self.num_bins = len(bin_edges) - 1
        self.log_num_bins = jnp.log(self.num_bins)

        # Calculate the distance of each observation from all bin centres so
        # that the shape is `(nobs, nbins)` and then turn it into a log-likelihood.
        dx = xobs[:, None] - self.bin_centres[None, :]

        self.ll_base = jnp.asarray(
            - 0.5 * (dx / xerr[:, None])**2 - 0.5 * jnp.log(2)
            - jnp.log(xerr)[:, None])

        self.ll_base -= jnp.log(
            gauss_intg(low_xobs, high_xobs, self.bin_centres[None, :],
                       xerr[:, None]))
        self._drch_arg = jnp.ones(self.num_bins)

    def __call__(self):
        log_bin_pdf = jnp.log(sample("bin_mass", Dirichlet(self._drch_arg)))

        ll = logsumexp(self.ll_base + log_bin_pdf[None, :], axis=-1)
        ll -= self.log_num_bins

        # To keep track of the likelihood values
        deterministic("ll_deterministic", jnp.sum(ll))

        factor("ll", ll)


###############################################################################
#                            Inference model                                  #
###############################################################################

model_kwargs = {"low_xobs": 0.09999999, "high_xobs": 20,}

model = DeltaFunctionModel(xobs, xerr, num_bins, **model_kwargs)

kernel = NUTS(model, init_strategy=init_to_median(num_samples=10000))
mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=True)
mcmc.run(rng_key)

mcmc.print_summary(exclude_deterministic=False)

samples = mcmc.get_samples()

ll = samples.pop("ll_deterministic")
ll_max = ll.max()
print(f"Maximum log-likelihood: {ll_max}")
nparams = samples["bin_mass"].shape[1]
BIC = nparams * np.log(len(xobs)) - 2 * ll_max
print(f"BIC: {BIC}")

# fname = "delta_"+str(num_bins)+".h5"
fname = "delta_"+str(num_bins)+".h5"        # _cut means that we have cut at age_err > 0.5
print(f"Saving to `{fname}.`")
with File(fname, "w") as f:
    grp = f.create_group("samples")
    for key, val in samples.items():
        grp.create_dataset(key, data=val)

    grp = f.create_group("data")
    grp.create_dataset("xobs", data=xobs)
    grp.create_dataset("xerr", data=xerr)
    # if xtrue is not None:
        # grp.create_dataset("xtrue", data=xtrue)

    grp = f.create_group("model")
    grp.create_dataset("bin_edges", data=model.bin_edges)
    grp.create_dataset("bin_centres", data=model.bin_centres)
    grp.create_dataset("bin_widths", data=model.bin_widths)

    grp = f.create_group("info")
    grp.create_dataset("ll", data=ll)
    grp.create_dataset("ll_max", data=ll_max)
    grp.create_dataset("BIC", data=BIC)


# Plot the reconstruction plot

bin_pdf = samples["bin_mass"] / model.bin_widths[None, :]

plt.figure()
plt.hist(model.xobs, bins="auto", label="Observations", density=1, zorder=0, histtype="step")
# if xtrue is not None:
#     plt.hist(xtrue, bins="auto", label="True", density=1, zorder=0,
#              histtype="step")

ylow, y, yhigh = np.percentile(bin_pdf, [16, 50, 84], axis=0)
plt.errorbar(model.bin_centres, y, yerr=[y - ylow, yhigh - y], capsize=2,
                label="Reconstruction")

plt.axvline(13.787, color="red", linestyle="--", label="Planck 2018")
plt.axvline(12.55734859, color="blue", linestyle="--", label="EDE (rough)")

plt.legend()
plt.xlim(np.min(xobs), np.max(xobs))
plt.xlabel(r"$\mathrm{Age} ~ [\mathrm{Gyr}]$")
plt.ylabel(r"Normalised PDF")

plt.tight_layout()
fname = "Plots/reconstruction_"+str(num_bins)+".png"
print(f"Saving reconstruction plot to `{fname}`.")
plt.savefig(fname, dpi=450)
plt.close()

# Now plot the corner plot
# keys = list(samples.keys())

# # Get labels and length of vector for each parameter
# labels = []
# nparam = np.zeros(len(keys), dtype=int)
# for i in range(len(keys)):
#     if len(samples[keys[i]].shape) == 1:
#         labels += [keys[i]]
#         nparam[i] = 1
#     else:
#         nparam[i] = samples[keys[i]].shape[1]
#         labels += [keys[i] + '_%i'%j for j in range(nparam[i])]

# nparam = [0] + list(np.cumsum(nparam))

# all_samples = np.empty((samples[keys[0]].shape[0], len(labels)))
# for i in range(len(keys)):
#     if len(samples[keys[i]].shape) == 1:
#         all_samples[:, nparam[i]] = samples[keys[i]][:]
#     else:
#         for j in range(nparam[i+1]-nparam[i]):
#             all_samples[:,nparam[i]+j] = samples[keys[i]][:,j]

# all_samples = np.array(all_samples)

# labels = [f"bin {i}" for i in range(model.num_bins)]

# plt.clf()
# corner.corner(all_samples, truths=None, labels=labels, show_titles=True, title_fmt='.3f', smooth=0, label_kwargs={"fontsize": 14}, title_kwargs={"fontsize": 14})
# fname = "Plots/corner_"+str(num_bins)+".png"
# print(f"Saving corner plot to `{fname}`.")
# plt.savefig(fname, dpi=450)

# plot_results(model, samples)
