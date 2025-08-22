
source_id,ipd_frac_multi_peak,ra,ra_error,dec,dec_error,parallax,parallax_error,parallax_over_error,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,bp_rp,phot_g_mean_flux_over_error,phot_rp_mean_flux_over_error,phot_bp_mean_flux_over_error,mass_flame,mass_flame_upper,mass_flame_lower,age_flame,age_flame_upper,age_flame_lower,classprob_dsc_combmod_binarystar





















#https://nadc.china-vo.org/res/r101467/

# A table of stellar age, abundance, and orbit action for 247,014 subgiant stars from LAMOST DR7 and Gaia eDR3
# The table contains 247,104 rows, 26 colums 
#---------------------------------------------------------------------------------------------------------------------------
# Column | Lines   | Label          |  Unit   |  Comments
#---------------------------------------------------------------------------------------------------------------------------
#   1    |   1-6   | number id      |         | 
#   2    |   7-20  | RA             |   deg   |  Right ascension (J2000) 
#   3    |  21-34  | Dec            |   deg   |  Declination (J2000)
#   4    |  35-42  | Age            |   Gyr   |
#   5    |  43-50  | Age unc.       |   Gyr   |  Uncertainty in age estimate
#   6    |  51-58  | [Fe/H]         |         |  LAMOST metallicity 
#   7    |  59-66  | [Fe/H] unc.    |         |  Uncertainty in [Fe/H]
#   8    |  67-77  | JR             | kpc.km/s|  Orbital radial action
#   9    |  78-88  | JPHI           | kpc.km/s|  Orbital azimuthal action
#  10    |  89-99  | JZ             | kpc.km/s|  Orbital vertical action
#  11    | 100-109 | X              |   kpc   |  X value in Galactic rectangular coordinate, positive to anti-center 
#  12    | 110-119 | Y              |   kpc   |  Y value in Galactic rectangular coordinate  
#  13    | 120-129 | Z              |   kpc   |  Z value in Galactic rectangular coordinate, positive to north pole
#  14    | 130-152 | Source id      |         |  Gaia eDR3 source id
#  15    | 153-192 | specid         |         |  LAMOST spectral id, in format of obsdate-planid-spectrograph-fiber
#  16    | 193-200 | S/N_g          |         |  LAMOST spectral S/N in SDSS g band
#  17    | 201-208 | Teff           |    K    |  LAMOST effective temperature 
#  18    | 209-216 | Teff unc.      |    K    |  Uncertainty in Teff
#  19    | 217-225 | logg           |  cm/s^2 |  LAMOST surface gravity derived  
#  20    | 226-234 | logg unc.      |   dex   |  Uncertainty in logg
#  21    | 235-243 | MK_spec        |   mag   |  Spectroscopic absolute magnitude in 2MASS Ks band 
#  22    | 244-252 | MK_spec unc.   |   mag   |  Uncertainty in MK_spec
#  23    | 253-261 | MK_comb        |   mag   |  Spectroscopic & geometric combined absolute magnitude in 2MASS Ks band
#  24    | 262-270 | MK_comb unc.   |   mag   |  Uncertainty in MK_comb
#  25    | 271-279 | [alpha/Fe]     |         |  LAMOST alpha-to-iron abundance ratio
#  26    | 280-288 | [alpha/Fe] unc.|         |  Uncertainty in [alpha/Fe]
#----------------------------------------------------------------------------------------------------------------------------



















ColDefs( 
   name = 'NUM'; format = 'K' 
   name = 'RAJ2000'; format = 'D' 
   name = 'DEJ2000'; format = 'D' 
   name = 'AGE'; format = 'E' 
   name = 'E_AGE'; format = 'E' 
   name = 'MASS'; format = 'E' 
   name = 'E_MASS'; format = 'E' 
   name = 'LOGDIS'; format = 'E' 
   name = 'E_LOGDIS'; format = 'E' 
   name = 'TEFF'; format = 'E' 
   name = 'E_TEFF'; format = 'E' 
   name = 'LOGG'; format = 'E' 
   name = 'E_LOGG'; format = 'E' 
   name = 'MK'; format = 'E' 
   name = 'E_MK'; format = 'E' 
   name = 'FEH'; format = 'E' 
   name = 'E_FEH'; format = 'E' 
   name = 'ALPHA_FE'; format = 'E' 
   name = 'E_ALPHA_FE'; format = 'E' 
   name = 'X'; format = 'E' 
   name = 'Y'; format = 'E' 
   name = 'Z'; format = 'E' 
   name = 'VR'; format = 'E' 
   name = 'VT'; format = 'E' 
   name = 'VZ'; format = 'E' 
   name = 'ECC'; format = 'E' 
   name = 'ENERGY'; format = 'E' 
   name = 'LZ'; format = 'E' 
   name = 'JR'; format = 'E' 
   name = 'JZ'; format = 'E' 
   name = 'R_GUIDING'; format = 'E' 
   name = 'R_APO'; format = 'E' 
   name = 'R_PERI'; format = 'E' 
   name = 'LAMOST_SPECID'; format = '36A' 
   name = 'LAMOST_SNR_G'; format = 'E' 
   name = 'VLOS'; format = 'E' 
   name = 'E_VLOS'; format = 'E' 
   name = 'EBV'; format = 'E' 
   name = 'E_EBV'; format = 'E' 
   name = 'MK_SPEC'; format = 'E' 
   name = 'E_MK_SPEC'; format = 'E' 
   name = 'GAIAEDR3_SOURCE_ID'; format = 'K' 
   name = 'PARALLAX'; format = 'E' 
   name = 'E_PARALLAX'; format = 'E' 
   name = 'PMRA'; format = 'E' 
   name = 'E_PMRA'; format = 'E' 
   name = 'PMDEC'; format = 'E' 
   name = 'E_PMDEC'; format = 'E' 
   name = 'RUWE'; format = 'E' 
   name = 'GAIA_G'; format = 'E' 
   name = 'E_GAIA_G'; format = 'E' 
   name = 'GAIA_BP'; format = 'E' 
   name = 'E_GAIA_BP'; format = 'E' 
   name = 'GAIA_RP'; format = 'E' 
   name = 'E_GAIA_RP'; format = 'E' 
   name = 'TWOMASS_J'; format = 'E' 
   name = 'E_TWOMASS_J'; format = 'E' 
   name = 'TWOMASS_H'; format = 'E' 
   name = 'E_TWOMASS_H'; format = 'E' 
   name = 'TWOMASS_K'; format = 'E' 
   name = 'E_TWOMASS_K'; format = 'E' 
)