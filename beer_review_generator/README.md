## Getting started

Apart from Keras used to do the neural network heavy lifting and the actual review generation, the main dependency in this project
is [ratebeer](https://github.com/OrganicIrradiation/ratebeer), a library designed specifically for the purpose of scraping data
from RateBeer. However, at the time I am writing this, the part of this library used to obtain beer reviews is not working.
As a workaround, I included a .patch file that you can apply to the ratebeer package source code once you download it.
The function featured in the patch worked for me, but as web APIs change frequently, it might not work for you now.

Once you got the ratebeer package up and working, run the reviews.py file to create two ~2MB files of programmatically cleaned up
reviews of IPAs and Stouts & Porters beers from RateBeer. With the text corpuses generated, you have everything it takes to train the RNN! But be prepared to wait for a looong time for the results - it really does take a lot of time to train this model.
If you don't have a GPU available, consider reducing the model's complexity significantly. 