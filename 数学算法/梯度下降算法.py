# new_weight = old_weight - learning_rate * gradient
def update_weights(self, learning_rate):
    for i in range(10):
        self.a_net[i] -= learning_rate * self.a_net_gradients[i]
        self.b_net[i] -= learning_rate * self.b_net_gradients[i]
        self.result_net[i] -= learning_rate * self.result_net_gradients[i]
