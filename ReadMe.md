# BCI Toolbox: Bayesian Causal Inference Toolbox

![Version](https://img.shields.io/badge/version-0.0.1.4-red)
![Language](https://img.shields.io/badge/language-Python-blue)

BCI Toolbox is a Python implementation of the hierarchical Bayesian Causal Inference (BCI) model for multisensory research. The BCI model is a statistical framework for understanding the causal relationships between sensory inputs and prior expectations of a common cause, which can account for human perception in various tasks.

## Installation

BCI Toolbox is available via pip. To install, run the following command:
```bash
python -m pip install bcitoolbox
```
Please check if the latest version is installed from [bcitoolbox](https://pypi.org/project/bcitoolbox).

## Mathematical Formulation

The BCI model is based on Bayes’ Rule as follows:

$P(C|xU, xD) = \frac{P(C) P(xU, xD|C)}{P(xU, xD)}$

where xU and xD are two signals received by the nervous system, and C is a binary variable denoting the number of causes in the environment, 1 or 2.


# BCI Toolbox: Basic Usage

## Graphical User Interface (GUI)

The BCI Toolbox provides a Graphical User Interface (GUI) for easy interaction with its functionalities. To utilize the GUI, you can follow these steps:

1. Import the BCI Toolbox package using the following code:

```python
import bcitoolbox as btb
```

2.  Call the GUI:
```python
btb.gui()
```

The current version GUI of the BCI toolbox consists of three main parts, which are ***Fitting module***, ***Simulation for continuous condition*** and ***Simulation for numerosity task***.

### ***Fitting module***
**1. Import** / **Open file**
   
Users can upload single or multiple files simultaneously via either ***Import*** or ***Open file***. Users can also add the file paths to the entry box and click ***Import*** to upload.

The selected files containing behavioral data must be **.csv** files and need to be in the following format:

| (True number of stimuli from modality U) | (True number of stimuli from modality D) | (Reported number of stimuli from modality U) | (Reported number of stimuli from modality D) |
|------------------------------|-----------------------------------------|--------------------------------------------|--------------------------------------------|
| ...                          | ...                                     | ...                                        | ...                                        |

**2. Number of simulations**
   
Number of samples for the probability distribution for each case. Users can choose 1000 for testing and 10000 for final publication.

**3. Fit type**
   
The BCI toolbox provides three fit types, which is also how errors are calculated:

***mll***: Minus log likelihood

***mr2***: Minus R square

***sse***: Sum of Squares for Errors

Users can select any one of it depending on specific condition.

**4. Decision Strategy**
   
The BCI toolbox provides three different decision strategies:

***Averaging***: model averaging

***Selecting***: model selection

***Matching***: probability matching

Users need to select at least one strategy for fitting. If selected strategies are more than one, the toolbox will automatically compare the results of each fit and output the optimal result.

**5. Parameters**
   
Users can set the target estimated parameters and set their ranges.

***pcommon***: The prior probability that both sensory information can be attributed to one cause.

***sigmaU***: The standard deviation of the Gaussian distribution of the likelihood for modality Up.

***sigmaD***: The standard deviation of the Gaussian distribution of the likelihood for modality Down.

***sigmap***: The standard deviation of the Gaussian distribution of the prior.

***mup***: The mean of the Gaussian distribution of the prior.

***sU***: A constant added to the mean of the Gaussian distribution for the likelihood for modality Up.

***sD***: A constant added to the mean of the Gaussian distribution for the likelihood for modality Down.

**6. Run**

Users can click ***run*** after the above steps and wait for the final results. The running status will be always updated on the page.

After the fitting is complete, the results of it will be presented in a new window. The user can browse the fitting results and click save to save the results as a **.txt** file.

**7. Plot**

Users can click ***plot*** to get the fitting result they want for a particular piece of data.

**8. Figure Save**

Users can click ***save*** to save all fitting figures to folder. 

**9. Main Page**

Go back to the main page.


### ***Simulation for continuous condition***

**1. Parameters**

See ‘***Fitting module***’ 5.

**2. Stimulus value**

***Stimulus U***: The true value of the stimulus (modality Up).

***Stimulus D***: The true value of the stimulus (modality Down).

**3. Elements**

***Response Distribution***: The s_hats of simulated responses, indicated by solid lines.

***Stimulus Encoding***: The encoding distributions under the assumption of Gaussian distribution, indicated by dotted lines.

***Prior Distribution***: The prior bias for the central location, indicated by green dotted line.

**3. Decision Strategy**

See ‘***Fitting module***’ 4.

**4. Estimates**
   
***Peak***: The peak of distributions, indicated by rhombuses.

***Mean***: The mean of distributions, indicated by inverted triangles.

***Display value***: Showing the value of the peak and mean of probability on the figure.

**6. Simulate**

Users can click ***simulate*** after the above steps and wait for the final results. 


### ***Simulation for numerosity task***

**1. Parameters**

See ‘***Fitting module***’ 5.

**2. Number of stimuli**

Number of unisensory condition.

**3. Decision Strategy**

See ‘***Fitting module***’ 4.

**4. Simulate**

Users can click ***simulate*** after the above steps and wait for the final results. 





## Authors

- **Haocheng (Evans) Zhu**
  
    **Email**: evanszhu2001@gmail.com

- **Dr. Ulrik R. Beierholm**
  
    **Email**: beierh@gmail.com

- **Dr. Ladan Shams**
  
    **Email**: ladan@psych.ucla.edu


If you have any inquiries or feedback, please don't hesitate to contact us. We will get back to you as soon as possible.

---














