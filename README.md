# Implementing Linear Regression with Gradient Descent from scratch with Python.

If two variables are correlated, if one increases, the other also does. Such relationship can be modeled/predicted using a line equation or **Linear Regression**.

$Y = m_{0} + mX$

While this works great for single variable, what if we want to predict / estimate an event that is dependent on multiple factors / vabriables / features? We would need to model multuple lines for each feature and construct a final equation of the target variable Y, involving bias and weights of each feature. So, the final equation function becomes - 

$Y = m_{0} + m_{0} x_{0} + m_{1} x_{1} + m_{ }x_{3} + ... + m_{n} x_{n}$

$Y = [ m_{0} + m_{1} + m_{2} + m_{3} + ... + m_{n} ] * [1; x_{1}; x_{2}; x_{3}; ... ; x_{n}]$ 

