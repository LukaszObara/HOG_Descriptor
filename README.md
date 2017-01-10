# HOG_Descriptor
For educational purposes we build a people detector as described in Navneet Dalal and Bill Triggs paper entitled <em>Histograms of Oriented Gradients for Human Detection</em>. We will be using the 160x96 pixel images found in <a href="http://pascal.inrialpes.fr/data/human/"> INRIA Person Dataset</a> to test our implementation. In addition we have provided a script to accomplish the same task using `OpenCV`. 

The method that we follow can be represented by the following diagram:
![](https://github.com/LukaszObara/HOG_Descriptor/blob/master/images/HOG_Steps.png "HOG steps").

We describe the process further in the <a href="https://github.com/LukaszObara/HOG_Descriptor/blob/master/HOG_Notebook.ipynb"> jupyter notebook. 

### OpenCV
Using the `OpenCV`:

![](https://github.com/LukaszObara/HOG_Descriptor/blob/master/images/People.png "Original")  ![](https://github.com/LukaszObara/HOG_Descriptor/blob/master/images/people_detect.png "Detection")

# References
<ol>
<li>Dalal Navneet, <em>Finding People in Images and Videos </em>(Ph.D. Thesis), retrived from http://lear.inrialpes.fr/people/dalal/NavneetDalalThesis.pdf</li>
<li>Dalal Navneet & Triggs Bill, <em>Histograms of Oriented Gradients for Human Detection</em>, CVPR, 2005, https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf</li>
</ol>
