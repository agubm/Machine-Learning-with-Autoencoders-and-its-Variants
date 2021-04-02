from time import time
from numpy import sqrt
import random
import matplotlib.pyplot as plt

N = 10000
EbNodB_range = range(0,20 )
itr = len(EbNodB_range)
ber = [None]*itr

for n in range (0, itr): 
 
    EbNodB = EbNodB_range[n]   
    EbNo=10.0**(EbNodB/10.0)
    noise_std = 1/sqrt(2*EbNo)
    noise_mean = 0
    
    no_errors = 0
    
    for m in range (0, N):
        tx_symbol1 = (2*random.randint(0,1)-1)
        tx_symbol2 = (2*random.randint(0,1)-1)
                
        noise1 = (random.gauss(noise_mean, noise_std)+
                1j*random.gauss(noise_mean, noise_std))
        noise2 = (random.gauss(noise_mean, noise_std)+
                1j*random.gauss(noise_mean, noise_std))
        
        
        ch_coeff1 = (random.gauss(0,1/sqrt(2))+
                    1j*random.gauss(0,1/sqrt(2)))
        ch_coeff2 = (random.gauss(0,1/sqrt(2))+
                    1j*random.gauss(0,1/sqrt(2)))
        
        rx_symbol1 =  ((1/sqrt(2))*tx_symbol1*ch_coeff1+ 
                       (1/sqrt(2))*tx_symbol2*ch_coeff2 + noise1)
        rx_symbol2 = (-(1/sqrt(2))*tx_symbol2*ch_coeff1+ 
                       (1/sqrt(2))*tx_symbol1*ch_coeff2 + noise2)
        
        estimate1 = (ch_coeff1.conjugate()*rx_symbol1+
                     ch_coeff2*rx_symbol2.conjugate())
        estimate2 = (ch_coeff2.conjugate()*rx_symbol1-
                     ch_coeff1*rx_symbol2.conjugate())
        
        
        
        
        
        det_symbol1 = 2*(estimate1.real >= 0) - 1
        det_symbol2 = 2*(estimate2.real >= 0) - 1
        
        no_errors += 1*(tx_symbol1 != det_symbol1)+1*(tx_symbol2 != det_symbol2)  
          
    ber[n] = 1.0*no_errors/(2*N)
    print ("EbNodB:", EbNodB)
    print ("Numbder of errors:", no_errors)
    print ("Error probability:", ber[n])