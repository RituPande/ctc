**Abstract:**  *Connectionist Temporal Classification* or CTC is a neural network output decoding and scoring algorithm that is used in sequence to sequence deep learning models. Sequence to sequence deep learning models take as input a sequence of length N and produce a output sequence of length M. CTC algorithm is used for those sequence to sequence models where M < N and the output symbols have the same order w.r.t. the input symbols i.e ouput symbols can be aligned to input seqeunce by repeating the same output symbol multiple times corresponding to a single input symbol, to form a label sequence. Label sequences are considered equivalent if they differ only in alignment. CTC scores are used with the back-propagation algorithm to update the neural network weights. This nature of CTC makes it ideal for speech recognition and Optical Character Recognition ( OCR ) tasks.
  
In this paper, we attempt to understand the principles and mathematics of *Connectionist Temporal Classiciation*. We also explore usage of CTC algorithm implemneted in Keras/Tensorflow library for breaking Captch 2.0.   

## 1. Sequence to Sequence Models  

Sequence to Sequence models are deep learning models that take a sequence of symbols as input and also output a seqeunce of symbols.  Sequence to Seqeunce models are implemented using Reccurrent Neural Networks or RNNs. RNNs take X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>N</sub> inputs at each time step respectively and output a label sequence Y<sub>1</sub>, Y<sub>2</sub>, ..., Y<sub>M</sub>. There are other variants of RNN which provide an output Y<sub>k</sub> as an additional input to step k+1. But for simplicity, we shall only focus on the most basic RNN implementation to understand the mechanics of CTC.  
  
  <p align="center">
  <img src="./images/RNN.png"> <br> 
  <b> Figure 1. Simple RNN </b>  
  </p> 

