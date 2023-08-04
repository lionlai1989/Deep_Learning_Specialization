# Sequence Models



## Description

This section provides a concise summary of each assignment in the course, accompanied by brief descriptions and a few figures.

- W1A1: [Building a Recurrent Neural Network - Step by Step](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C5-Sequence_Models/W1A1-Building_a_Recurrent_Neural_Network_Step_by_Step/Building_a_Recurrent_Neural_Network_Step_by_Step.html)  
  I implement a basic RNN and LSTM model with only **Numpy**. No other machine learning framework is used.
  
  <figure float="left">
  <img src="./W1A1-Building_a_Recurrent_Neural_Network_Step_by_Step/images/LSTM_figure4_v3a.png" height="300"/>
  <figcaption style="font-size: small;">An LSTM network comprises three gates, which can keep the information flowing from the current time step to the next, mitigating the vanishing gradient problem.</figcaption>
  </figure>

- W1A2: [Character level language model - Dinosaurus Island](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C5-Sequence_Models/W1A2-Dinosaur_Island_Character_Level_Language_Modeling/Dinosaurus_Island_Character_level_language_model.html)  
  A language model is a probability model designed to represent the language domain. Here, I use 1536 dinosaur names to train a character-level language model to generate artificial names for new dinosaurs. For instance, `Eiaantoe` would probably be a good name for a newly found dinosaur.

- W1A2: [Improvise a Jazz Solo with an LSTM Network](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C5-Sequence_Models/W1A3-Improvise_a_Jazz_Solo_with_an_LSTM_NetworkImprovise_a_Jazz_Solo_with_an_LSTM_Network_v4.html)  
  An improvised Jazz solo is generated. You can hear the music for yourself.
  [Click here to download the MP3 file](./W1A3-Improvise_a_Jazz_Solo_with_an_LSTM_Network/output/rendered.wav)



- W1A2: [](https://htmlpreview.github.io/?)  

## Reference:

- Week 1:
  - [Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086) (GitHub: karpathy)
  - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy blog, 2015)
  - [deepjazz](https://github.com/jisungk/deepjazz) (GitHub: jisungk)
  - [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf) (Gillick, Tang & Keller, 2010)
  - [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf) (Keller & Morrison, 2007)
  - [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf) (Pachet, 1999)

- Week 2:
  - [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) (Bolukbasi, Chang, Zou, Saligrama & Kalai, 2016)
  - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) (Pennington, Socher & Manning, 2014)
  - [Woebot](https://woebothealth.com/)

- Week 4:
  - [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?) (by DeepLearning.AI)
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, 2017)
