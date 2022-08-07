<a name="readme-top"></a>
# Intelligent-topological-engineering-of-quantum-p-magnets




<!-- ABOUT THE PROJECT -->
## About The Project


This github is for the project: Intelligent topological engineering of single-molecule quantum π-magnets 

We have built a deep learning infrastructure to realize atomically precise multi-bond surgery of single-molecule quantum π-magnets. The deep learning infrastructure consists of several deep learning models and can communicate with the instrument smoothly via the labview based control system.

The details of the deep learning models and the control system are demonstrated inside the paper. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

<img width="1034" alt="image" src="https://user-images.githubusercontent.com/44763317/183239091-daabfdec-5729-47d6-b08f-eb86c8be6a8a.png">


The whole program is running in labview. The labview is acting as a control center. During the molecular surgery, it will collect data from the instrument and send to the AI modules. After getting feedbacks from the AI modules, it will control the instrument to conduct certain operations. Then the data is collected again from the instrument to the labview. The whole system forms a closed operation loop.         

* Labview

The configuration of the labview is important. Please ensure inside each python node, the location of the python script is updated to your file location. In addition please ensure to set the port to communicate with the instrument correctly.

`
Check the port number of your nanonis control software and make sure inside labview the port number is set as the same.
`

* Deep Learning Modules for Molecular Surgery

The model resources for the deep learning models is in the below link:

`
https://drive.google.com/drive/folders/1RqPAVn20Smia9943qSStBg4cJwq45_3e?usp=sharing
`

Please ensure all the python scripts and model resourses are in the same folder. It is for your simplicity since then you don't need to change the path.


* Training the deep learning modules

The training scripts of the deep learning modules is in the train folder. The models are developed through pytorch.The labelled training data is provided in the link below:

`
https://drive.google.com/drive/folders/1RqPAVn20Smia9943qSStBg4cJwq45_3e?usp=sharing
`

Running example for deep learning modules can be found [here](Running_example/DL_modules)


* Shap analysis

The shap analysis is to interpret our deep learning modules. It is instance-based explaination. It will analysis the feature importance learned by the intelligent modules on a certain sample.

All the codes for the shap analysis inside our project is in the Shap folder. The details of the implementation is in the paper.

Running example for shap analysis can be found [here](Running_example/Shap)


## Pre-requirements
Python ≥ 3.6     
https://www.python.org/ but install conda is recommended, due to it provides more packages in build, you don’t have to install packages one by one.

Pytorch ≥ 1.4    
https://pytorch.org/ click get stated, select your preference, and run the install command based on your working environment.

Numpy ≤ 1.17   
Install by using pip install or conda. Don’t install NumPy 1.18 because it cannot safely interpret flaot64 as an integer. You’ll receive this kind of error when evaluating.

Install detectron2, imgaug, and labelIMG. Install detectron2 on Windows is quite time-consuming. Mac OS is recommended if you got one. If you want to install detectron2 on Windows, please refer to useful link.

Install detectron2: https://github.com/facebookresearch/detectron2

Install imgaug: https://github.com/aleju/imgaug

Install labelIMG: https://github.com/tzutalin/labelImg

Install shap analysis package via pip install shap or conda install -c conda-forge shap

Install Labview via https://www.ni.com/en-sg/support/downloads/software-products/download.labview.html#460283

Install the Lavbiew-Nanonis programming interface via https://www.specs-group.com/nc/nanonis/products/detail/programming-interface/


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Li jiali - lijiali1994@gmail.com

Yin jun - yinjun_56@qq.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Su jie who is the leading first of our paper

<p align="right">(<a href="#readme-top">back to top</a>)</p>



