import numpy as np
from pade import pade
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def process_signal(t,signal):

    # Scipy FFT
    SIGMA = 5  # control amount of lineshape broadening (bigger =  more narrow)
    dt = t[1] - t[0]
    damp = np.exp(-(dt*np.arange(len(signal)))/float(SIGMA))

    fw_fft = fft(signal*damp)
    freq_fft = fftfreq(t.size,d=dt)*2*np.pi  # 2pi is implicit in Pade, so we add it here
    
    # grab the positive frequency components
    fw_fft = fw_fft[:t.size//2]
    freq_fft = freq_fft[:t.size//2]
    
    # Now do Pade appx of Fourier Transform; line broadening via SIGMA
    # we will read in frequencies from SciPy FFT to evaluate analytic function over (read_freq)
    fw_pade, freq_pade = pade(t,signal,sigma=SIGMA,read_freq=freq_fft)
    
    # grab just imaginary components
    fw_pade = np.imag(fw_pade)
    fw_fft = np.imag(fw_fft)

    return fw_pade, freq_pade, fw_fft, freq_fft

def test_pade(): 
    w  = 2.0
    dt = 0.02
    N  = 5000
    t  = np.linspace(0,dt*N, N, endpoint=False)
    signal = np.sin(w*t) + np.sin(2*w*t) + np.sin(4*w*t) 

    fw_pade, freq_pade, fw_fft, freq_fft = process_signal(t,signal)

    # FFT and Pade agree to within some tolerance
    assert np.allclose(freq_pade,freq_fft)
    assert np.linalg.norm(fw_fft - fw_pade) < 1e-4  # "closeness" also depends on signal length, SIGMA, etc. 
    #print(np.linalg.norm(fw_fft - fw))

if __name__ == '__main__':

    w  = 2.0
    dt = 0.02
    N  = 5000
    t  = np.linspace(0,dt*N, N, endpoint=False)
    signal = np.sin(w*t) + np.sin(2*w*t) + np.sin(4*w*t) 

    fw_pade, freq_pade, fw_fft, freq_fft = process_signal(t,signal)

    plt.plot(t,signal)
    plt.savefig('signal.png')
    plt.clf()
    plt.cla()

    plt.plot(freq_pade,fw_pade,label='pade')
    plt.plot(freq_fft,fw_fft,label='FFT',ls='--')
    plt.xlim([0,10])
    plt.legend()
    plt.savefig('fsignal.png')
    plt.show()


