class Neuron { 
    constructor(numInputs) { 
        this.weights = []; 
        this.bias = Math.random(); 
        for (let i = 0; i < numInputs; i++) { 
            this.weights.push(Math.random()); 
        } 
    } 

    activate(sum) { 
        return 1 / (1 + Math.exp(-sum));
    } 

    feedforward(inputs) { 
        let sum = 0; 
        for (let i = 0; i < this.weights.length; i++) { 
            sum += inputs[i] * this.weights[i]; 
        } 
        sum += this.bias; 
        return this.activate(sum); 
    } 

    train(inputs, error, learningRate) { 
        let output = this.activate(this.bias + this.weights.reduce((sum, weight, i) => sum + weight * inputs[i], 0)); 
        output *= 1 - output; 
        let delta = learningRate * error * output; 
        this.bias += delta; 
        for (let i = 0; i < this.weights.length; i++) { 
            this.weights[i] += delta * inputs[i]; 
        } 
    }
}
class Layer { 
    constructor(numNeurons, numInputs) { 
        this.neurons = []; 
        for (let i = 0; i < numNeurons; i++) { 
            this.neurons.push(new Neuron(numInputs)); 
        } 
    } 

    feedforward(inputs) { 
        let outputs = []; 
        for (let neuron of this.neurons) { 
            outputs.push(neuron.feedforward(inputs)); 
        } 
        return outputs; 
    }

    train(inputs, errors, learningRate) { 
        for (let i = 0; i < this.neurons.length; i++) { 
            this.neurons[i].train(inputs, errors[i], learningRate); 
        } 
    }
}
class NeuralNetwork { 
    constructor(numInputs, numHidden1, numHidden2, numOutput) { 
        this.inputLayer = new Layer(numInputs, numInputs); 
        this.hiddenLayer1 = new Layer(numHidden1, numInputs); 
        this.hiddenLayer2 = new Layer(numHidden2, numHidden1); 
        this.outputLayer = new Layer(numOutput, numHidden2); 
    } 

    feedforward(inputs) { 
        let hidden1Outputs = this.hiddenLayer1.feedforward(this.inputLayer.feedforward(inputs)); 
        let hidden2Outputs = this.hiddenLayer2.feedforward(hidden1Outputs); 
        let outputs = this.outputLayer.feedforward(hidden2Outputs); return outputs; 
    } 

    backpropagate(inputs, targets, learningRate) { 
        let hidden1Outputs = this.hiddenLayer1.feedforward(this.inputLayer.feedforward(inputs)); 
        let hidden2Outputs = this.hiddenLayer2.feedforward(hidden1Outputs); 
        let outputs = this.outputLayer.feedforward(hidden2Outputs); 
        // Вычисляем ошибку выходного слоя и обновляем веса перед ним 
        let outputErrors = []; 
        for (let i = 0; i < this.outputLayer.neurons.length; i++) { 
            outputErrors.push(targets[i] - outputs[i]); 
        } 
        let hidden2Errors = []; 
        for (let i = 0; i < this.hiddenLayer2.neurons.length; i++) { 
            let error = 0; 
            for (let j = 0; j < this.outputLayer.neurons.length; j++) { 
                error += outputErrors[j] * this.outputLayer.neurons[j].weights[i]; 
            } 
            hidden2Errors.push(error); 
        } 
        let hidden1Errors = []; 
        for (let i = 0; i < this.hiddenLayer1.neurons.length; i++) { 
            let error = 0; 
            for (let j = 0; j < this.hiddenLayer2.neurons.length; j++) { 
                error += hidden2Errors[j] * this.hiddenLayer2.neurons[j].weights[i]; 
            } 
            hidden1Errors.push(error); 
        } 
        this.outputLayer.train(hidden2Outputs, outputErrors, learningRate); 
        this.hiddenLayer2.train(hidden1Outputs, hidden2Errors, learningRate); 
        this.hiddenLayer1.train(this.inputLayer.feedforward(inputs), hidden1Errors, learningRate); 
    }
}

//--------------------------------------------------------------------------------------------------------------

let nn = new NeuralNetwork(2, 3, 2, 1);
// Обучение
for (let i = 0; i < 100000; i++) { 
    nn.backpropagate([0, 0], [0], 0.1); 
    console.log("[0, 0] 0"); 
    console.log(nn.feedforward([0, 0])); 

    nn.backpropagate([0, 1], [1], 0.1); 
    console.log("[0, 1] 1"); 
    console.log(nn.feedforward([0, 1])); 

    nn.backpropagate([1, 0], [1], 0.1); 
    console.log("[1, 0] 1"); 
    console.log(nn.feedforward([1, 0])); 

    nn.backpropagate([1, 1], [0], 0.1);
    console.log("[1, 1] 0"); 
    console.log(nn.feedforward([1, 1])); 
}
console.log(); 
    // Тестирование
    console.log("Тестирование"); 
    console.log(nn.feedforward([0, 0])); 
    // Ожидаемый результат: [0]
    console.log(nn.feedforward([0, 1])); 
    // Ожидаемый результат: [1]
    console.log(nn.feedforward([1, 0])); 
    // Ожидаемый результат: [1]
    console.log(nn.feedforward([1, 1])); 
    // Ожидаемый результат: [0]