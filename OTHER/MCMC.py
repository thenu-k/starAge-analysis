# As _final but using binned PDF model rather than Gaussian

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
from scipy.stats import norm

from jax import config
config.update("jax_enable_x64", True)

def rebin_pdf(pdf, old_edges, new_edges):
    """
    Rebin a PDF from old_edges to new_edges.
    pdf: shape (n_stars, n_old_bins)
    old_edges: shape (n_old_bins+1,)
    new_edges: shape (n_new_bins+1,)
    Returns: shape (n_stars, n_new_bins)
    """
    n_stars, n_old_bins = pdf.shape
    n_new_bins = len(new_edges) - 1
    rebinned = np.zeros((n_stars, n_new_bins))
    # For each new bin, sum the overlapping portions of the old bins
    for i in range(n_new_bins):
        left, right = new_edges[i], new_edges[i+1]
        # Find overlap of [left, right] with each old bin
        for j in range(n_old_bins):
            old_left, old_right = old_edges[j], old_edges[j+1]
            overlap_left = max(left, old_left)
            overlap_right = min(right, old_right)
            overlap = max(0, overlap_right - overlap_left)
            if overlap > 0:
                frac = overlap / (old_right - old_left)
                rebinned[:, i] += pdf[:, j] * frac
    # Normalize each row to sum to 1
    rebinned /= rebinned.sum(axis=1, keepdims=True)
    return rebinned


plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

num_bins = int(10)

# gen = np.random.default_rng(42)
# rng_key = random.PRNGKey(42)

gen = np.random.default_rng(np.random.randint(0, 100000000))
rng_key = random.PRNGKey(np.random.randint(0, 100000000))

nwarm, nsamp = 500, 2500

arr = np.load('ageData.npy')
age, age_err = arr[:, 0], arr[:, 1]

mask = (age<20)&(age>0) & (age_err>0.001)

print("Initial number of stars:", len(age), flush=True)

age, age_err = age[mask], age_err[mask]

print("Final number of stars:", len(age), flush=True)

# xobs, xerr = age, age_err

# Define bins
# bin_edges = np.arange(0, 20.01, 0.2)  # edges from 0 to 20, step 0.2
# bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# n_bins = len(bin_centres)

# 80 bins from 0.1 to 8.0 (inclusive) in steps of 0.1
edges1 = np.arange(0.1, 8.1, 0.1)  # 0.1, 0.2, ..., 8.0

# 60 bins from 8.2 to 20.0 (inclusive) in steps of 0.2
edges2 = np.arange(8.2, 20.2, 0.2)  # 8.2, 8.4, ..., 20.0

# Combine, making sure to avoid duplicate edge at 8.0/8.2
bin_edges = np.concatenate([edges1, edges2])
# Ensure the first edge is 0.0 and last is 20.0 
bin_edges = np.insert(bin_edges, 0, 0.0)
bin_edges[-1] = 20.0

bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
n_bins = len(bin_centres)

N = len(age)

# Compute the CDF at each bin edge for each star
# shape: (N, n_bins+1)
# cdf_edges = norm.cdf(bin_edges[None, :], loc=age[:, None], scale=age_err[:, None])

# The probability in each bin is the difference between CDFs at the bin edges
# star_bin_pdfs = cdf_edges[:, 1:] - cdf_edges[:, :-1]  # shape: (N, n_bins)

# Normalize each row to sum to 1 (in case of numerical error)
# star_bin_pdfs /= star_bin_pdfs.sum(axis=1, keepdims=True)

N = len(age)
# Compute the CDF at each bin edge for each star
cdf_edges = norm.cdf(bin_edges[None, :], loc=age[:, None], scale=age_err[:, None])
star_bin_pdfs = cdf_edges[:, 1:] - cdf_edges[:, :-1]  # shape: (N, n_bins)
star_bin_pdfs = np.load("PDFs-2.npy")  
star_bin_pdfs /= star_bin_pdfs.sum(axis=1, keepdims=True)

# Save for use in your model
# np.save("stellar_bin_pdfs.npy", star_bin_pdfs)
# np.savetxt("stellar_bin_pdfs.dat", star_bin_pdfs)

# star_bin_pdfs = np.load("stellar_bin_pdfs.npy")           # FROM THE REAL DATA
# shape (n_stars, n_bins), where star_bin_pdfs[i, j] is the probability for star i in bin j.

n_stars = star_bin_pdfs.shape[0]
print(f"Number of stars: {n_stars}", flush=True)

# star_bin_edges = np.linspace(0, 20, 101)
recon_bin_edges = np.linspace(0, 20, num_bins+1)
# recon_bin_centres = 0.5 * (recon_bin_edges[:-1] + recon_bin_edges[1:])

star_bin_pdfs_rebinned = rebin_pdf(star_bin_pdfs, bin_edges, recon_bin_edges)

###############################################################################
#                          Delta function model                               #
###############################################################################

# def gauss_intg(a, b, mu, sigma):
#     return 0.5 * (
#         + erf((b - mu) / (jnp.sqrt(2) * sigma))
#         - erf((a - mu) / (jnp.sqrt(2) * sigma))
#         )

# class DeltaFunctionModel:

#     def __init__(self, xobs, xerr, num_bins, low_xobs, high_xobs,
#                  bin_edges=None):
#         self.xobs = xobs
#         self.xerr = xerr

#         if bin_edges is not None:
#             if xobs.min() < bin_edges[0] or xobs.max() > bin_edges[-1]:
#                 raise ValueError(
#                     "Some values are outside of the binning range.")
#         else:
#             xmin, xmax = 0, xobs.max()
#             print(f"Using {num_bins} bins from {xmin} to {xmax}.", flush=True)
#             bin_edges = jnp.linspace(xmin, xmax, num_bins + 1)

#         self.bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
#         self.bin_edges = bin_edges
#         self.bin_widths = bin_edges[1:] - bin_edges[:-1]
#         self.log_bin_widths = jnp.log(self.bin_widths)
#         self.num_bins = len(bin_edges) - 1
#         self.log_num_bins = jnp.log(self.num_bins)

#         # Calculate the distance of each observation from all bin centres so
#         # that the shape is `(nobs, nbins)` and then turn it into a log-likelihood.
#         dx = xobs[:, None] - self.bin_centres[None, :]

#         self.ll_base = jnp.asarray(
#             - 0.5 * (dx / xerr[:, None])**2 - 0.5 * jnp.log(2) - jnp.log(xerr)[:, None])

#         self.ll_base -= jnp.log(
#             gauss_intg(low_xobs, high_xobs, self.bin_centres[None, :], xerr[:, None]))
#         self._drch_arg = jnp.ones(self.num_bins)

#     def __call__(self):
#         log_bin_pdf = jnp.log(sample("bin_mass", Dirichlet(self._drch_arg)))

#         ll = logsumexp(self.ll_base + log_bin_pdf[None, :], axis=-1)
#         ll -= self.log_num_bins

#         # To keep track of the likelihood values
#         deterministic("ll_deterministic", jnp.sum(ll))

#         factor("ll", ll)

class BinnedPDFModel:
    # def __init__(self, star_bin_pdfs, num_bins):
    #     self.star_bin_pdfs = jnp.asarray(star_bin_pdfs)  # shape (n_stars, n_bins)
    #     self.num_bins = num_bins
    #     self._drch_arg = jnp.ones(self.num_bins)

    def __init__(self, star_bin_pdfs, num_bins, bin_edges=None):
        self.star_bin_pdfs = jnp.asarray(star_bin_pdfs)  # shape (n_stars, n_bins)
        self.num_bins = num_bins
        self._drch_arg = jnp.ones(self.num_bins)
        if bin_edges is not None:
            self.bin_edges = jnp.asarray(bin_edges)
            self.bin_centres = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
            self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        else:
            self.bin_edges = None
            self.bin_centres = None
            self.bin_widths = jnp.ones(self.num_bins)

    def __call__(self):
        # bin_mass is the population-level PDF over bins (to infer)
        bin_mass = sample("bin_mass", Dirichlet(self._drch_arg))  # shape (num_bins,)
        log_bin_mass = jnp.log(bin_mass)  # shape (num_bins,)

        # For each star, the likelihood is sum_k (star_pdf_k * bin_mass_k)
        # (This is a mixture: the probability of the star's PDF under the population PDF)
        # Compute log-likelihood for each star:
        # log(sum_k star_bin_pdfs[i, k] * bin_mass[k])
        ll = jnp.log(jnp.sum(self.star_bin_pdfs * bin_mass[None, :], axis=1))

        deterministic("ll_deterministic", jnp.sum(ll))
        factor("ll", ll)


###############################################################################
#                            Inference model                                  #
###############################################################################

model_kwargs = {"low_xobs": 0.09999999, "high_xobs": 20,}

# model = DeltaFunctionModel(xobs, xerr, num_bins, **model_kwargs)
# model = BinnedPDFModel(star_bin_pdfs, num_bins)
# model = BinnedPDFModel(star_bin_pdfs_rebinned, num_bins)
model = BinnedPDFModel(star_bin_pdfs_rebinned, num_bins, bin_edges=recon_bin_edges)

kernel = NUTS(model, init_strategy=init_to_median(num_samples=10000))
mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=True)
mcmc.run(rng_key)

mcmc.print_summary(exclude_deterministic=False)

samples = mcmc.get_samples()

ll = samples.pop("ll_deterministic")
ll_max = ll.max()
print(f"Maximum log-likelihood: {ll_max}")
nparams = samples["bin_mass"].shape[1]
BIC = nparams * np.log(n_stars) - 2 * ll_max
print(f"BIC: {BIC}")

# fname = "delta_"+str(num_bins)+".h5"
fname = "delta_"+str(num_bins)+"_BINNED.h5"        # _cut means that we have cut at age_err > 0.5
print(f"Saving to `{fname}.`")

# with File(fname, "w") as f:
#     grp = f.create_group("samples")
#     for key, val in samples.items():
#         grp.create_dataset(key, data=val)

#     grp = f.create_group("data")
#     grp.create_dataset("xobs", data=model.xobs)
#     grp.create_dataset("xerr", data=xerr)
#     # if xtrue is not None:
#         # grp.create_dataset("xtrue", data=xtrue)

#     grp = f.create_group("model")
#     grp.create_dataset("bin_edges", data=model.bin_edges)
#     grp.create_dataset("bin_centres", data=model.bin_centres)
#     grp.create_dataset("bin_widths", data=model.bin_widths)

#     grp = f.create_group("info")
#     grp.create_dataset("ll", data=ll)
#     grp.create_dataset("ll_max", data=ll_max)
#     grp.create_dataset("BIC", data=BIC)


# Plot the reconstruction plot

bin_pdf = samples["bin_mass"] / model.bin_widths[None, :]

plt.figure()
# plt.hist(model.xobs, bins="auto", label="Observations", density=1, zorder=0, histtype="step")
plt.hist(age, bins="auto", label="Observations", density=1, zorder=0, histtype="step")
# if xtrue is not None:
#     plt.hist(xtrue, bins="auto", label="True", density=1, zorder=0,
#              histtype="step")

ylow, y, yhigh = np.percentile(bin_pdf, [16, 50, 84], axis=0)
plt.errorbar(model.bin_centres, y, yerr=[y - ylow, yhigh - y], capsize=2, label="Reconstruction")

plt.axvline(13.787, color="red", linestyle="--", label="Planck 2018")
plt.axvline(12.55734859, color="blue", linestyle="--", label="EDE (rough)")

plt.legend()
# plt.xlim(np.min(xobs), np.max(xobs))
plt.xlabel(r"$\mathrm{Age} ~ [\mathrm{Gyr}]$")
plt.ylabel(r"Normalised PDF")

plt.tight_layout()
fname = "Plots/reconstruction_"+str(num_bins)+"_BINNED_2.png"
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
