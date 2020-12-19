

from pydub import AudioSegment
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import soundfile as sf
from scipy.io.wavfile import read
from flask import send_file


# Math operations and 
import math
import functools
import scipy.special as spsp
from scipy.special import exp1

# File handling 
import glob, os

# Data handling 
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
# Deep learning: Modelling helpers
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.python.ops.signal import window_ops
from tensorflow.keras.layers import Activation, Add, Conv1D, Conv2D, Dense, Dropout,Flatten, LayerNormalization, MaxPooling2D, ReLU, Input, Masking

#app = Flask(__name__)

app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run


try:
  os.remove("uploads/converted.wav")
except:
  pass

# all deepxi codes

#hemasunder
# Functions for audio handling
def save_wav(path, wav, f_s):
    """"
    save_wav: Save the denoised audio to the given path
        path                  - path to save the output audio file 
        f_s                   - sampling freq
        wav                   - audio file extension
        np.squeeze            - Remove single-dimensional entries from the shape of an array
        if block(isinstance)  - function returns True if the specified object is of the specified type, otherwise False 
                                checking if the file is float dtype
        np.asarray            - convert an given input to an array
     """
    wav = np.squeeze(wav)
    if isinstance(wav[0], np.float32):
        wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16) 
    sf.write(path, wav, f_s)

def read_wav(path):
    """
    read_wav: Read the audio files from a given path
    """
    wav, f_s = sf.read(path, dtype='int16')
    return wav, f_s

def batch(fdir):
    """
    batch: Create a bacth of input audio files
        fdir           - Input path of audio files 
        fname_l        - list of file names.
        wav_l          - list of fetched_audio files
        len_l          - list of audio file lengths
    Returns: 
    audio_files_numpy_format, np array of audio file legnths, list of file_names
    """
    fname_l = [] 
    wav_l = [] 
    fnames = ['*.wav', '*.flac', '*.mp3']

    #get all the supported sound file types from the given path
    for fname in fnames:  
        for fpath in glob.glob(os.path.join(fdir, fname)): 
            (wav, _) = read_wav(fpath) # read each audio file using the soundfile library 
            if np.isnan(wav).any() or np.isinf(wav).any():
                raise ValueError('Error: NaN or Inf value. File path: %s.' % (fdir))
            wav_l.append(wav) #add the wavefile name to the wav_l list
            fname_l.append(os.path.basename(os.path.splitext(fpath)[0])) # append respective path 

    len_l = []
    # get maximum audio length among all the files, so that all the other files are padded with zeroes to have a uniform batch
    maxlen = max(len(wav) for wav in wav_l) 
    wav_np = np.zeros([len(wav_l), maxlen], np.int16) # creating a numpy array of zeroes, with the length of the largest audio file as a dimension 

    for (i, wav) in zip(range(len(wav_l)), wav_l):
      #Overlapping the zeros in the numpy array of zeroes, to create a padded array for the smaller files to make them equal in size to large files
        wav_np[i,:len(wav)] = wav 
        len_l.append(len(wav))
    return wav_np, np.array(len_l, np.int32),fname_l



#CUSTOM CLASS FOR DIGITAL SIGNAL PROCESSING 
var = 0
varout = 0

class STFT:
    """
    Short-Term Fourier Transform:
        N_d                -  window duration (samples)
        N_s                -  window shift (samples)
        NFFT               -  number of DFT componts [ Discrete fourier transform]
        f_s                - sampling freq
    """
    #Mohit
    def __init__(self, N_d, N_s, NFFT, f_s):
        self.N_d = N_d
        self.N_s = N_s
        self.NFFT = NFFT
        self.f_s = f_s
        self.W = functools.partial(window_ops.hamming_window, periodic=False) # A callable that takes a window length and returns a [window_length] Tensor of samples in the provided datatype.
        self.ten = tf.cast(10.0, tf.float32) # Casting the tensor to the float32 type
    #Mohit
    def polar_analysis(self, x):
        """
        x                  -  Input numpy array
        tf.signal.stft     -  Computes the Short-time Fourier Transform of signals
        Returns:
        tf.abs             -  computes absolute value of tensor         
        tf.math.angle      -  returns element wise arguments of a complex tensor
      """
        STFT = tf.signal.stft(x, self.N_d, self.N_s, self.NFFT, window_fn=self.W, pad_end=True) # Find STFT of a given signal
        return tf.abs(STFT), tf.math.angle(STFT) # Returns magnitude and phase angle of resulting STFT
    #Rohan
    def polar_synthesis(self, STMS, STPS):
        """
        tf.cast                      - Casts a tensor to a new type
        tf.complex                   - A Tensor of type complex64 or complex128
        tf.exp                       - performs exponential operation on a tensor
        
        Returns:
        tf.signal.inverse_stft       - inverse the stft input signals
        STMS                         - short time magnitude spectrum
        STPS                         - short time phase spectrum
        """
        STFT = tf.cast(STMS, tf.complex64)*tf.exp(1j*tf.cast(STPS, tf.complex64)) 
        return tf.signal.inverse_stft(STFT, self.N_d, self.N_s, self.NFFT, tf.signal.inverse_stft_window_fn(self.N_s, self.W))

class DeepXiInput(STFT):
    def __init__(self, N_d, N_s, NFFT, f_s, mu=None, sigma=None):
        """
        defining mu and sigma
        """
        super().__init__(N_d, N_s, NFFT, f_s)
        self.mu = mu 
        self.sigma = sigma
    #Nobody        
    def observation(self, x):
        """
        Returns STMS and STPS from the given input numpy (converted audio file)
        """
        x = self.normalise(x)
        x_STMS, x_STPS = self.polar_analysis(x)
        return x_STMS, x_STPS
    


    # Defining all functions required for processing audio file after converting it to tensors using tensor mathematical operations. 
    #Shubham
    def normalise(self, x):
        #normailzation / standardization
        """
        Normalize the given input np array  
        """
        return tf.truediv(tf.cast(x, tf.float32), 32768.0) # Divides x tensor by y elementwise
    #Shubham
    def n_frames(self, N):
        """
        tf.math.ceil - Return the ceiling of the input, element-wise
        """
        return tf.cast(tf.math.ceil(tf.truediv(tf.cast(N, tf.float32), tf.cast(self.N_s, tf.float32))), tf.int32)


    #Surya
    def xi_hat(self, xi_bar_hat):
        """
        scipy.special.erfinv(y)  -  Inverse of the gause error function erf.
        """ 
        xi_db_hat = np.add(np.multiply(np.multiply(self.sigma, np.sqrt(2.0)),
                                       spsp.erfinv(np.subtract(np.multiply(2.0, xi_bar_hat), 1))), self.mu)
        return np.power(10.0, np.divide(xi_db_hat, 10.0))



# MMSE-LSA gain function.
#Purnasai
def gfunc(xi, gamma=None):
    """
    MMSE-LSA Gain function
    """
    nu = np.multiply(np.divide(xi, np.add(1, xi)), gamma)
    G = np.multiply(np.divide(xi, np.add(1, xi)), np.exp(np.multiply(0.5, exp1(nu))))
    return G


# Modelling ResNet architecture

class ResNet:
    """
    ResNet: Residual Neural Network - Base model for DeepXi architeture
    """
    def __init__(self,inp,n_outp,n_blocks,d_model,d_f,k,max_d_rate,padding,):
        self.d_model = d_model
        self.d_f = d_f
        self.k = k
        self.n_outp = n_outp
        self.padding = padding
        self.first_layer = self.feedforward(inp)
        self.layer_list = [self.first_layer]
        for i in range(n_blocks): self.layer_list.append(self.block(self.layer_list[-1], int(2**(i%(np.log2(max_d_rate)+1)))))
        self.logits = Conv1D(self.n_outp, 1, dilation_rate=1, use_bias=True)(self.layer_list[-1])
        self.outp = Activation('sigmoid')(self.logits)     

    # 1st layer
    def feedforward(self, inp):
        """
        1D convolution layer (temporal convolution)   -    This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) 
                                                            dimension to produce a tensor of outputs.

        """
        ff = Conv1D(self.d_model, 1, dilation_rate=1, use_bias=False)(inp)
        norm = LayerNormalization(axis=2, epsilon=1e-6)(ff)
        act = ReLU()(norm)
        return act

    # 2nd layer
    def block(self, inp, d_rate):
        """
        2D convolution layer (spatial convolution)
        """
        self.conv_1 = self.unit(inp, self.d_f, 1, 1, False)
        self.conv_2 = self.unit(self.conv_1, self.d_f, self.k, d_rate,
            False)
        self.conv_3 = self.unit(self.conv_2, self.d_model, 1, 1, True)
        residual = Add()([inp, self.conv_3])
        return residual

    # 3rd layer
    def unit(self, inp, n_filt, k, d_rate, use_bias):
        """
        dilation_rate                     : an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
                                          Can be a single integer to specify the same value for all spatial dimensions. 
        use_bias                          : Boolean, whether the layer uses a bias vector.
        Relu                              : Clips value in range of 0 to infinity , so clips all negative value to zero
        normalization layer               :  Normalize the activations of the previous layer for each given example in a batch independently,
                                           rather than across a batch like Batch Normalization. i.e. applies a transformation that maintains the mean activation within each example close to 0
                                             and the activation standard deviation close to 1.
        """
        norm = LayerNormalization(axis=2, epsilon=1e-6)(inp)
        act = ReLU()(norm)
        conv = Conv1D(n_filt, k, padding=self.padding, dilation_rate=d_rate,
            use_bias=use_bias)(act)
        return conv


class DeepXi(DeepXiInput):
    def __init__(self,N_d,N_s,NFFT,f_s,model_path,stat_path,**kwargs):  
        super().__init__(N_d, N_s, NFFT, f_s)
        self.n_feat = math.ceil(self.NFFT/2 + 1)
        self.n_outp = self.n_feat
        self.inp = Input(name='inp', shape=[None, self.n_feat], dtype='float32')
        self.mask = Masking(mask_value=0.0)(self.inp) 
        #Masking is a way to tell sequence-processing layers that certain timesteps in an input are missing, and thus should be skipped when processing the data.
        
        self.network = ResNet(
            inp=self.mask,
            n_outp=self.n_outp,
            n_blocks=kwargs['n_blocks'],
            d_model=kwargs['d_model'],
            d_f=kwargs['d_f'],
            k=kwargs['k'],
            max_d_rate=kwargs['max_d_rate'],
            padding=kwargs['padding'],
            )
        #Padding is done in order to make all sequences in a batch fit a given standard length.
        self.model = Model(inputs=self.inp, outputs=self.network.outp)
        self.model.summary()
        # The Actual program starts from this line
        self.sample_stats(stat_path) # Load sample statistics file to derive mu and sigma values
        self.model.load_weights(model_path) #Load Weights of Saved_model from model_path


    def infer(self,test_x,test_x_len,test_x_base_names,out_path='out/denoised/',n_filters=40,):
        
        out_path = out_path # setting output directory 
        
        print("Processing observations...")
        x_STMS_batch, x_STPS_batch, n_frames = self.observation_batch(test_x, test_x_len) # observation_batch is function defined at the last part of the cell

        print("Performing inference...")
        xi_bar_hat_batch = self.model.predict(x_STMS_batch, batch_size=1, verbose=1)# MAX TIME TAKEN
        #purnasai
        batch_size = len(test_x_len)  # taking length of x_batch
        for i in tqdm(range(batch_size)):  # Module for iterating batches # here batch size is 1 so it will iterate only one time.
            base_name = test_x_base_names[i]  #this and below are all the mathematical operations done for audio processing and getting dezire output as per domain understanding.
            x_STMS = x_STMS_batch[i,:n_frames[i],:] 
            x_STPS = x_STPS_batch[i,:n_frames[i],:]
            xi_bar_hat = xi_bar_hat_batch[i,:n_frames[i],:]
            xi_hat = self.xi_hat(xi_bar_hat)
            
            y_STMS = np.multiply(x_STMS, gfunc(xi_hat, xi_hat+1))
            y = self.polar_synthesis(y_STMS, x_STPS).numpy()
            save_wav(out_path+ base_name + '.wav', y, self.f_s)
    #purnasai
    def sample_stats(self,stats_path='data/'): # loading sample stats present in sample stats folder as a zip file # existing stats files, required for following operations.
        if os.path.exists(stats_path + 'stats.npz'):
            print('Loading sample statistics...')
            with np.load(stats_path + 'stats.npz') as stats:
                self.mu = stats['mu_hat']   # getting value for mu from the stats file
                self.sigma = stats['sigma_hat'] #getting value for sigma from the stats file
                

    #Rohan
    def observation_batch(self, x_batch, x_batch_len):
        """
        batch_size       - getting size of numpy (converted audio)
        max_n_frames     - taking maximum value of array size
        x_STMS_batch     - create numpy of zeros to have equal sized arrays across the batch
        n_feat           - 
        STMS             - short time magnitude spectrum
        STPS             - short time phase spectrum
        """
        batch_size = len(x_batch) # taking length of x_batch
        max_n_frames = self.n_frames(max(x_batch_len)) # getting nframe value for input x_batch_len
        x_STMS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32) # creating a numpy with zero values but of desire shape as per dimension of batch size,max_n_frames
        x_STPS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32) # creating a numpy with zero values but of desire shape as per dimension of batch size,max_n_frames
        n_frames_batch = [self.n_frames(i) for i in x_batch_len]  # getting nframe value for value in x_batch_len
        for i in tqdm(range(batch_size)): # Module for iterating batches # here batch size is 1 so it will iterate only one time.
            x_STMS, x_STPS = self.observation(x_batch[i,:x_batch_len[i]])  # this and below are the mathematical operations done for audio processing and getting dezire output as per domain understanding.
            x_STMS_batch[i,:n_frames_batch[i],:] = x_STMS.numpy() 
            x_STPS_batch[i,:n_frames_batch[i],:] = x_STPS.numpy()
        return x_STMS_batch, x_STPS_batch, n_frames_batch



#VARIABLES FOR THE MODEL
d_model  = 256     #block output size
n_blocks  = 40     #no of blocks in the model
d_f = 64           #block bottlekneck size
k =  3             #convolution kernel size
max_d_rate = 16    #max_dilation_rate
padding = "causal" #type of convnet padding
f_s  = 16000       #sampling frequency
T_d  = 32          #window duration
T_s  =  16         #window shift
#rohan
N_d = int(f_s*T_d*0.001) # window duration (samples).
N_s = int(f_s*T_s*0.001) # window shift (samples).
NFFT = int(pow(2, np.ceil(np.log2(N_d)))) # number of DFT components.

# end of deepxi codes





def log_request_info(request):
    app.logger.warning("request.path: {0}".format(request.path))
    app.logger.warning("request.files: {0}".format(request.files))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        sound = AudioSegment.from_file(f)
        #sound = AudioSegment.from_wav(f)
        sound = sound.set_channels(1)
        sound.export("uploads/converted.wav", format="wav")
        print(sound)
        # start deepxi
        # PATH VARIABLE
        data_path='data/' # Path of the sample_stats file that to be loaded for inference purpose.
        test_x_path='uploads/' # Path of the inputs : noisy audio files
        out_path='denoised/' # Path to which out the output audio file is saved.
        model_path='variables/variables' # Path of the TF Saved_model
        test_x, test_x_len, test_x_base_names = batch(test_x_path) # Fetch the test noisy audio inputs along with its names.
        # DeepXi object instantiation
        deepxi = DeepXi(N_d=N_d,N_s=N_s,NFFT=NFFT,f_s=f_s,d_model=d_model,n_blocks=n_blocks,d_f=d_f,k=k,max_d_rate =max_d_rate,padding=padding
                        ,model_path=model_path,stat_path=data_path) 
        deepxi.infer(test_x=test_x,test_x_len=test_x_len,test_x_base_names=test_x_base_names,out_path= out_path) 
        path_to_file = "uploads/converted.wav"
        return send_file(path_to_file,mimetype="audio/wav",as_attachment=True, attachment_filename="converted.wav")
    return None



app.run()