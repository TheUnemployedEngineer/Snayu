
def predict(network ,inputs):
    output=inputs
    for layer in network:
        output=layer.forward(output)    #every layers output is passing into the next layers input 
    
    return output
