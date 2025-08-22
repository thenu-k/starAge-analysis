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

# import os
# os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4'


def MCMC_ANALYSIS(mask=None, bins=50, counts=[500, 2500], PDF_link='OTHER\\PDFs-2.npy', writeTo='test'):
    print('-> Running MCMC analysis')
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

    plt.clf()
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    num_bins = int(bins)  

    gen = np.random.default_rng(np.random.randint(0, 100000000))
    rng_key = random.PRNGKey(np.random.randint(0, 100000000))

    [nwarm, nsamp] = counts

    arr = np.load('OTHER\\ageData.npy')
    age, age_err = arr[0], arr[1]  
    edges1 = np.arange(0.1, 8.1, 0.1)  
    edges2 = np.arange(8.2, 20.2, 0.2)  
    bin_edges = np.concatenate([edges1, edges2])
    bin_edges = np.insert(bin_edges, 0, 0.0)
    bin_edges[-1] = 20.0

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centres)
    N = len(age)
    cdf_edges = norm.cdf(bin_edges[None, :], loc=age[:, None], scale=age_err[:, None])
    star_bin_pdfs = cdf_edges[:, 1:] - cdf_edges[:, :-1]  # shape: (N, n_bins)
    star_bin_pdfs = np.load(PDF_link) 
    star_bin_pdfs /= star_bin_pdfs.sum(axis=1, keepdims=True)
    initialAgeDistr = star_bin_pdfs.sum(axis=0)
    ageGrid = np.load('OTHER\\ageGrid.npy')
    if mask is not None:
        star_bin_pdfs = star_bin_pdfs[mask]
        
    n_stars = star_bin_pdfs.shape[0]
    print(f"Number of stars: {n_stars}", flush=True)

    recon_bin_edges = np.linspace(0, 20, num_bins+1)

    star_bin_pdfs_rebinned = rebin_pdf(star_bin_pdfs, bin_edges, recon_bin_edges)


    class BinnedPDFModel:
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

    model = BinnedPDFModel(star_bin_pdfs_rebinned, num_bins, bin_edges=recon_bin_edges)
    # return [None, None, [model.bin_centres, None]]
 
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=10000))
    mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=True)
    mcmc.run(rng_key)

    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()

    ll = samples.pop("ll_deterministic")
    # print(ll)
    ll_max = ll.max()
    print(f"Maximum log-likelihood: {ll_max}")
    nparams = samples["bin_mass"].shape[1]
    BIC = nparams * np.log(n_stars) - 2 * ll_max
    print(f"BIC: {BIC}")



    # Plot the reconstruction plot

    bin_pdf = samples["bin_mass"] / model.bin_widths[None, :]

    plt.figure()

    ylow, y, yhigh = np.percentile(bin_pdf, [16, 50, 84], axis=0)
    plt.errorbar(model.bin_centres, y, yerr=[y - ylow, yhigh - y], capsize=2, label="Reconstruction")

    plt.axvline(13.787, color="red", linestyle="--", label="Planck 2018")
    plt.axvline(12.55734859, color="blue", linestyle="--", label="EDE (rough)")

    plt.legend()
    # plt.xlim(np.min(xobs), np.max(xobs))
    plt.xlabel(r"$\mathrm{Age} ~ [\mathrm{Gyr}]$")
    plt.ylabel(r"Normalised PDF")

    plt.tight_layout()
    fname = "OUTPUT\\W5\\reconstruction_"+str(writeTo)+"_BINNED_2.png"
    print(f"Saving reconstruction plot to `{fname}`.")
    plt.savefig(fname, dpi=450)
    plt.close()

    return [samples, ll, [model.bin_centres, y]]


