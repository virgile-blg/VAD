import numpy as np
import librosa


def add_wgn(s,var=1e-4):
    """
        Add white Gaussian noise to signal
        If no variance is given, simply add jitter. 
        Jitter helps eliminate all-zero values.
        """
    np.random.seed(0)
    noise = np.random.normal(0,var,len(s))
    return s + noise


def read_wav(filename):
    """
        read wav file.
        Normalizes signal to values between -1 and 1.
        Also add some jitter to remove all-zero segments."""
    #x, sr = wavfile.read(filename) # scipy reads int
    x, sr = librosa.load(filename, sr=16000)
    x = np.array(x)/float(max(abs(x)))
    x = add_wgn(x) # Add jitter for numerical stability
    return sr,x

#===============================================================================
def enframe(x, win_len, hop_len):
    """
        receives a 1D numpy array and divides it into frames.
        outputs a numpy matrix with the frames on the rows.
        """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
    x_framed = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        x_framed[i] = x[i * hop_len : i * hop_len + win_len]
    return x_framed


def deframe(x_framed, win_len, hop_len):
    '''
        interpolates 1D data with framed alignments into persample values.
        This function helps as a visual aid and can also be used to change 
        frame-rate for features, e.g. energy, zero-crossing, etc.
        '''
    n_frames = len(x_framed)
    n_samples = n_frames*hop_len + win_len
    x_samples = np.zeros((n_samples,1))
    for i in range(n_frames):
        x_samples[i*hop_len : i*hop_len + win_len] = x_framed[i]
    return x_samples


import librosa
import numpy as np


##Function definitions:
def vad_help():
    """Voice Activity Detection (VAD) tool.
	
	Navid Shokouhi May 2017.
    """
    print("Usage:")
    print("python unsupervised_vad.py")

#### Display tools
def plot_this(s,title=''):
    """
     
    """
    import pylab
    s = s.squeeze()
    if s.ndim ==1:
        pylab.plot(s)
    else:
        pylab.imshow(s,aspect='auto')
        pylab.title(title)
    pylab.show()

def plot_these(s1,s2):
    import pylab
    try:
        # If values are numpy arrays
        pylab.plot(s1/max(abs(s1)),color='red')
        pylab.plot(s2/max(abs(s2)),color='blue')
    except:
        # Values are lists
        pylab.plot(s1,color='red')
        pylab.plot(s2,color='blue')
    pylab.legend()
    pylab.show()


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes,axis=1)
    xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
    return xframes

def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes,xframes.T))/float(n_frames)

def compute_log_nrg(xframes):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes,axis=1)
    X = np.abs(X[:,:X.shape[1]/2])**2
    return np.sqrt(X)


def nrg_vad(xframes,percent_thr,nrg_thr=0.,context=5):
    """
        Picks frames with high energy as determined by a 
        user defined threshold.
        
        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding. 
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.
        
        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        """
    xframes = zero_mean(xframes)
    n_frames = xframes.shape[0]
    
    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames,1))
    for i in range(n_frames):
        start = max(i-context,0)
        end = min(i+context,n_frames-1)
        n_above_thr = np.sum(xnrgs[start:end]>nrg_thr)
        n_total = end-start+1
        xvad[i] = 1.*((float(n_above_thr)/n_total) > percent_thr)
    return xvad


# ==============================================================================

def plot_nrg_vad(test_file, percent_high_nrg=0.0):
    fs,s = read_wav(test_file)
    win_len = int(fs*0.025)
    hop_len = int(fs*0.025)
    sframes = enframe(s,win_len,hop_len)

    vad = 1 - nrg_vad(sframes,percent_high_nrg)

    plot_these(deframe(vad,win_len,hop_len),s)


def get_non_speech(file, percent_high_nrg=0.001):
    fs,s = read_wav(file)
    win_len = int(fs*0.025)
    hop_len = int(fs*0.025)
    sframes = enframe(s,win_len,hop_len)

    vad = 1 - nrg_vad(sframes,percent_high_nrg)

    return s, vad