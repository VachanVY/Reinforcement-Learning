# HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION

* ![image](https://github.com/user-attachments/assets/d282198f-edfe-4a8a-b980-d900c4524f3d)

> * Bias means the estimates systematically differ from the true expected return. 
This happens when you use bootstrapping (i.e., estimating value using other learned estimates) 
like in actor-critic methods or TD learning
> * Variance is caused by Monte-Carlo updates, there are many paths from one point to another, which may result in different returns which may very a lot

* ![image](https://github.com/user-attachments/assets/54e43f5d-3a4a-4ce8-a4ea-67a7d12b4a07)
* ![image](https://github.com/user-attachments/assets/ae71380b-182e-49dd-8601-37c78ad4695e)
* ![image](https://github.com/user-attachments/assets/6fcb5a20-5628-4f6a-a644-32f63a5054e4)
> * We will introduce a parameter γ that allows us to reduce variance by downweighting rewards corresponding to delayed effects, at the cost of introducing bias. This parameter corresponds to the
discount factor used in discounted formulations of MDPs, but we treat it as a variance reduction
parameter in an undiscounted problem 
* ![image](https://github.com/user-attachments/assets/ffa75b79-6e0a-4f24-9855-77cb8b068b8c)
* ![image](https://github.com/user-attachments/assets/4ae88482-633d-4c57-9ac9-66a06daa0ec9)
* $$A^{GAE}_t = \sum_{i=1}\gamma^{i-1}\lambda^{i-1}\delta_{t+i-1}$$
* ![image](https://github.com/user-attachments/assets/0006ad97-79dc-418d-bfc6-510d43c95039)

## Eligibilty Traces
* ![image](https://github.com/user-attachments/assets/b89da5de-b9f7-49fb-82f1-593786f07228)
* ![image](https://github.com/user-attachments/assets/c41f0b8f-27b1-4d80-9f63-9f30947b9d8f)
* ![image](https://github.com/user-attachments/assets/c1ea04d1-a4a5-4654-8163-f81158b21437)
* ![image](https://github.com/user-attachments/assets/eb96d863-c089-4aa4-89c5-34644fa165f6)
* ![image](https://github.com/user-attachments/assets/54d2f8d7-5124-41cb-9aeb-373d3adea156)
* ![image](https://github.com/user-attachments/assets/e02fd2a1-e45f-4e1c-a796-86425bbbb345)
* ![image](https://github.com/user-attachments/assets/ed66fae7-c9df-4654-a98f-b1f4a982b3c2)
