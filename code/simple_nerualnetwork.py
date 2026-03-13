import math
import random


#激活函数 ,把数字归一化到 0到1范围
def sigmoid(x):
    if x > 100:
        return 1
    elif x < -100:
        return 0
    else:
        return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

class SimpleNeuron:
    def __init__(self, num_inputs):
        #随机权重
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        #偏置
        self.bias = random.uniform(-1, 1)
        self.output = 0

    # 前向推理，进行预测，得出结果。
    def feedforward(self, inputs):
        #加权和
        totol =  sum(w * x for w, x in zip(self.weights, inputs))
        #偏置
        totol += self.bias
        #激活
        self.output = sigmoid(totol)
        return self.output

    # 后向推理，训练模型,使其调整参数(权重，偏置)
    def train(self, inputs, target, learning_rate = 0.1):
        error = target - self.output

        # 导数越大需要调整力度越大
        delta = error * sigmoid_derivative(self.output)

        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * delta * inputs[i];

        self.bias += learning_rate * delta

        return abs(error)

def generate_training_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        y = random.uniform(0, 10)
        x = random.uniform(0, 10)
        label = 1 if y > x else 0
        data.append(([x, y], label))
    return data

def visualize_decision(neuron, test_points):
    print("\n Testing the trained neuron: ")
    print('-' * 70)
    print(f"{'Point':<15} | {'Prediction':<15} | {'Actual':<15} | {'Correct?'}")
    print("-" * 70)

    correct = 0
    for point, actual in test_points:
        prediction = neuron.feedforward(point)
        predicted_class = 1 if prediction > 0.5 else 0
        actual_class = actual
        is_correct = "v" if predicted_class == actual_class else "x"

        if predicted_class == actual_class:
            correct+=1

        print(f"({point[0]:5.2f}, {point[1]:5.2f}) | {prediction:14.4f} | {actual_class:^15} | {is_correct}")

    print('-'*70)
    accuracy = (correct / len(test_points)) * 100
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_points)} correct)")


def main():
    # 一个简单的神经网络  ,判断点是否在 y=x 的上面或下面
    print("="*70)

    #1.生成训练数据
    training_data = generate_training_data(100)

    # 显示部分训练数据
    print("\n显示部分训练数据:")
    for i in range(3):
        point, label = training_data[i]
        position = "上" if label == 1 else "下"
        print(f"  点 ({point[0]:.2f}, {point[1]:.2f}) 在线 y=x {position}")

    #2.
    neuron = SimpleNeuron(2)
    print(f"Initial weights: [{neuron.weights[0]:.3f}, {neuron.weights[1]:.3f}]")
    print(f"Initial bias: {neuron.bias:.3f}")

    #3.训练
    epochs = 50
    for epoch in range(epochs):
        total_error = 0

        for inputs, target in training_data:
            neuron.feedforward(inputs)
            error = neuron.train(inputs, target, 0.1)
            total_error += error

        # Show progress
        if (epoch + 1) % 10 == 0:
            avg_error = total_error / len(training_data)
            print(f"Epoch {epoch + 1}/{epochs} - Average error: {avg_error:.4f}")

    print("\n✅ Training complete!")
    print(f"Final weights: [{neuron.weights[0]:.3f}, {neuron.weights[1]:.3f}]")
    print(f"Final bias: {neuron.bias:.3f}")

    # Step 4: Test the neuron
    test_data = generate_training_data(num_samples=10)
    visualize_decision(neuron, test_data)


if __name__ == "__main__":
    main()