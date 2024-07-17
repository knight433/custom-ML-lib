import numpy as np 

class svm:

    def __init__(self) -> None:
        self.w = None
        self.b = None

    def slove(self,x,y,learningRate=0.01,lambda_para = 0.01,epcoh=1000):
        
        n_samples, n_features = x.shape
        
        y_ = np.where(y <= 0,-1,1) # assigning -ve or +ve part of the plains 

        self.w = np.random.rand(n_features)
        self.b = 0 

        for _ in range(epcoh):
            for i, x_i in enumerate(x):

                sideOfPlane = y_[i] * (np.dot(x_i,self.w) - self.b) >= 1
                
                if sideOfPlane == True:
                    self.w = self.w - (learningRate * (2*lambda_para*self.w) )
                
                else:
                    self.w = self.w - (learningRate * (2 * lambda_para * self.w - np.dot(x_i,y_[i])))
                    self.b = self.b - (learningRate * y_[i] )

        print(f'weights = {self.w} bias = {self.b}')
