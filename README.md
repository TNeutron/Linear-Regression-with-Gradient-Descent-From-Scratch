#Multivariate Linear Regression with Gradient Descent from Scratch using Python


If you are here already, you already know what linear regression is and only care about implementation. So, lets run down ASAP.


## Theory Rundown

### Notations -

Machine learning community, with their weird choices of notations, make the equations look more complicated than they are. 
Following notations will be used here -

$H$= Hypothesis Equation
<br>$J$  = Cost Function
<br>$X$ = Variables / Features / Inputs 
<br>$W_i/\theta_i$ = Weights for each Variables / Features / Inputs 
<br>$W_0/\theta_0$ = Bias 
<br>$\frac{\delta J}{\delta \theta_i}$ = Gradient of a the i-th variable/input/feature
<br>$\alpha$ = Learning Rate
<br>$m$ = Total Number of Variables / Features / Inputs 
<br>$n$ = Total Number of Samples

## It starts with `hypothesis function`
The function/equation by which we model the relationship between output and inputs ( $\theta_n$, n being the number of features/variables ). It’s output is your model’s prediction - 

$$
H(\theta) = \theta_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_n*X_n
$$

We are predicting/estimating/plotting one output value based on the contribution (weights $\theta_i$) of all other variables.<br>


## But… But… But… How do I find the optimal weight values? Using Gradient Descent!
We start with random values for each weight and iteratively approach toward an optimal value. 

```python
W = np.random.randn(1, n)
```

Gradient descent works in 4 easy steps -

1. Calculate predicted value with the randomly generated initial weight values - 

$$
H(\theta) = \theta_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_n*X_n
$$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0663fd79-2654-44f0-b6c8-185271311dca/47b5c288-1637-4b93-ba0c-1745545ce41e/Untitled.png)

In code - 

```python
theta_0 = 0 # or b or bias set to 0 in the beginning

# Assume Training_x variable contains numpy dataframe of your training samples
# W[0] required to access row from generated 2D matrix  
H_func = theta_0 + np.dot(self.training_x, W[0]) 
```

1. Calculate the sum of error using an appropriate cost function, Mean Square Error in this case.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0663fd79-2654-44f0-b6c8-185271311dca/f8e67583-1e7e-4ed3-a527-1ba95e769902/Untitled.png)

EQ - 

$$
Cost Function = \frac{1}{\text{Total Number of Features(m)}} * \text{Sum}(Actual - Predicted)^2 
$$

----
$$ J(\theta_i) = \frac{1}{m} \sum_{i=1}^{m} ((Y_{i}) - H(\theta_i) )^2 $$ 

Breaking down the $H(\theta_i)$ makes it - 

$$
J(\theta_i) = \frac{1}{m} \sum_{i=1}^{m} (Y_{i} - \theta_0 - \theta_1X_1 - \theta_2X_2 - ... - \theta_m*X_m)^2
$$

```python
j = np.sum(((H_func - np.array(train_y) )**2.0)) / (m)
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0663fd79-2654-44f0-b6c8-185271311dca/0ffd9376-1ff8-4590-9ea6-b20b38adac01/Untitled.png)

1. Find Gradients for each weight by differentiating the cost function with each weight -

$$
\frac{\delta J}{\delta \theta_i} = \frac{2}{m} \sum_{i=1}^{n} (Y_i - H(\theta_i))*X_{i}
$$

For example, Gradient for $\theta_1$ will be -

$$
\frac{\delta J}{\delta \theta_1} = \frac{2}{m} \sum_{i=1}^{n} (Y_i - H(\theta_i))*X_{1}
$$

```python
# The common term of the derivative until dot multiplication of Xi
d_H_func = ((2/m) * (H_func - np.array(self.training_y)))

# Gradient of all the weights
d_W = np.dot(d_Y_hat, self.training_x)

# Gradient of the bias term (W_0 / Theta_0)
d_theta_0  = np.sum(d_Y_hat)
```

1. Update the weights by subtracting the gradient multiplied by learning rate -

$$
\text{Updated Weight, }\theta_{updated} = \theta_{old} - \frac{\delta J}{\delta \theta_1} * \alpha
$$

- **Positive derivative means** - Cost in respect to Weight is increasing. Decreasing the weight, moves us towards a lower cost.
- **Negative derivative means -** Cost in respect to Weight is decreasing. Increasing the weight will move us towards a lower cost.

```python
W = W - (learning_rate * d_W)
theta_0  = theta_0 - (learning_rate * d_theta_0 )
self.W = W
self.theta_0 = theta_0  
```

---

## Repeating these 4 steps, for enough times will result in the most optimal values of the weights and bias values by minimizing cost/error in the process until things “stabilize”.

Plotting the cost over number of iteration give us cost function.
