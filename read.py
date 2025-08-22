from DUMP.GLOBAL import GLOBAL
GLOBAL = GLOBAL()
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
hdul = fits.open('Stellar_ages.fits')
data = hdul[1].data
# print(data[1])
age = data['AGE']
e_age = data['E_AGE']       # Similarly for any other column you want to extract

# percentageErrors = e_age/age * 100

print('hello')

file = open('allColumns.txt', 'w')
for i in range(len(age)):
    for j in range(len(GLOBAL.COLUMNS)):
        file.write(str(GLOBAL.COLUMNS[j])+ ':----'+ str(data[GLOBAL.COLUMNS[j]][i])  + ';END;   ')
        os.system('cls')
        print('Row: '+ str(i))  
        print('Column: '+ str(j))
    file.write('%\n')
file.close()

# plt.hist(age, bins='auto')
# plt.savefig('ageHistogram.png')

# plt.hist(percentageErrors, bins='auto')
# plt.savefig('errorHistogram.png')

