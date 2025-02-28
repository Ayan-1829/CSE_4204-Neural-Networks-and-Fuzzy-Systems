x = X[i]
        x = np.insert(x, 0, 1);
        weighted_sum = np.dot(x, weights)
        
        output = 1 if weighted_sum >= 0 else 0
        
        delta = Y[i] - output
        
        if Y[i]==0 and output==1:
            weights -= learning_rate*x
        elif Y[i]==1 and output==0:
            weights += learning_rate*x