-- Define example questions and expected answers
local training_data = {
    {question = "have you been doing good", answer = 1},
    {question = "are you happy", answer = 1},
    {question = "are you sad", answer = 0},
    {question = "is it raining", answer = 0},
    {question = "do you like lua", answer = 1},
    {question = "is it sunny", answer = 1},
    {question = "do you hate bugs", answer = 0},
    {question = "is it night", answer = 0},
    {question = "is it cool being a AI", answer = 1},
    {question = "do you use any libarys" answer = 0},
    {question = "are you doing good" answer = 1},
    {question = "are you good" answer = 1}
}

-- Build vocabulary and convert questions to numerical format
local function build_vocabulary(data)
    local vocabulary = {}
    local index = 1
    for _, item in ipairs(data) do
        for word in item.question:gmatch("%w+") do
            if not vocabulary[word] then
                vocabulary[word] = index
                index = index + 1
            end
        end
    end
    return vocabulary
end

local function question_to_tensor(question, vocabulary, input_size)
    local tensor = {}
    for i = 1, input_size do
        tensor[i] = 0
    end
    for word in question:gmatch("%w+") do
        local index = vocabulary[word]
        if index then
            tensor[index] = 1
        end
    end
    return tensor
end

local vocabulary = build_vocabulary(training_data)
local input_size = #vocabulary

-- Define the neural network
local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

function NeuralNetwork.new(input_size, hidden_size, output_size)
    local self = setmetatable({}, NeuralNetwork)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    -- Initialize weights and biases
    self.weights_input_hidden = {}
    self.weights_hidden_output = {}
    self.bias_hidden = {}
    self.bias_output = {}

    -- Randomly initialize weights
    math.randomseed(os.time())
    for i = 1, input_size * hidden_size do
        self.weights_input_hidden[i] = math.random() * 2 - 1
    end
    for i = 1, hidden_size * output_size do
        self.weights_hidden_output[i] = math.random() * 2 - 1
    end
    for i = 1, hidden_size do
        self.bias_hidden[i] = math.random() * 2 - 1
    end
    for i = 1, output_size do
        self.bias_output[i] = math.random() * 2 - 1
    end

    return self
end

-- Sigmoid activation function
local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

-- Derivative of the sigmoid function
local function sigmoid_derivative(x)
    return x * (1 - x)
end

-- Forward pass
function NeuralNetwork:forward(inputs)
    self.inputs = inputs

    -- Hidden layer
    self.hidden = {}
    for i = 1, self.hidden_size do
        self.hidden[i] = 0
        for j = 1, self.input_size do
            self.hidden[i] = self.hidden[i] + inputs[j] * self.weights_input_hidden[(i - 1) * self.input_size + j]
        end
        self.hidden[i] = sigmoid(self.hidden[i] + self.bias_hidden[i])
    end

    -- Output layer
    self.outputs = {}
    for i = 1, self.output_size do
        self.outputs[i] = 0
        for j = 1, self.hidden_size do
            self.outputs[i] = self.outputs[i] + self.hidden[j] * self.weights_hidden_output[(i - 1) * self.hidden_size + j]
        end
        self.outputs[i] = sigmoid(self.outputs[i] + self.bias_output[i])
    end

    return self.outputs
end

-- Backward pass and training
function NeuralNetwork:train(inputs, targets, learning_rate)
    -- Forward pass
    self:forward(inputs)

    -- Calculate output layer errors and deltas
    local output_errors = {}
    local output_deltas = {}
    for i = 1, self.output_size do
        output_errors[i] = targets[i] - self.outputs[i]
        output_deltas[i] = output_errors[i] * sigmoid_derivative(self.outputs[i])
    end

    -- Calculate hidden layer errors and deltas
    local hidden_errors = {}
    local hidden_deltas = {}
    for i = 1, self.hidden_size do
        hidden_errors[i] = 0
        for j = 1, self.output_size do
            hidden_errors[i] = hidden_errors[i] + output_deltas[j] * self.weights_hidden_output[(j - 1) * self.hidden_size + i]
        end
        hidden_deltas[i] = hidden_errors[i] * sigmoid_derivative(self.hidden[i])
    end

    -- Update weights and biases
    for i = 1, self.output_size do
        for j = 1, self.hidden_size do
            self.weights_hidden_output[(i - 1) * self.hidden_size + j] = self.weights_hidden_output[(i - 1) * self.hidden_size + j] + learning_rate * output_deltas[i] * self.hidden[j]
        end
        self.bias_output[i] = self.bias_output[i] + learning_rate * output_deltas[i]
    end
    for i = 1, self.hidden_size do
        for j = 1, self.input_size do
            self.weights_input_hidden[(i - 1) * self.input_size + j] = self.weights_input_hidden[(i - 1) * self.input_size + j] + learning_rate * hidden_deltas[i] * self.inputs[j]
        end
        self.bias_hidden[i] = self.bias_hidden[i] + learning_rate * hidden_deltas[i]
    end
end

-- Function to predict yes or no
function NeuralNetwork:predict(inputs)
    local outputs = self:forward(inputs)
    return outputs[1] > 0.5 and "Yes" or "No"
end

-- Convert training questions to tensors
local inputs = {}
local labels = {}
for _, item in ipairs(training_data) do
    table.insert(inputs, question_to_tensor(item.question, vocabulary, input_size))
    table.insert(labels, {item.answer})
end

-- Create and train the neural network
local net = NeuralNetwork.new(input_size, 3, 1)
local learning_rate = 0.1
local epochs = 10000

for epoch = 1, epochs do
    for i = 1, #inputs do
        net:train(inputs[i], labels[i], learning_rate)
    end
    if epoch % 1000 == 0 then
        local loss = 0
        for i = 1, #inputs do
            local output = net:forward(inputs[i])
            loss = loss + (labels[i][1] - output[1]) ^ 2
        end
        loss = loss / #inputs
        print(string.format("Epoch: %d, Loss: %.4f", epoch, loss))
    end
end

-- Function to preprocess and predict user questions
local function ask_questions()
    while true do
        print("Ask a question (or type 'exit' to quit):")
        local question = io.read()

        if question == "exit" then
            break
        end

        local input_tensor = question_to_tensor(question, vocabulary, input_size)
        local prediction = net:predict(input_tensor)
        print("Answer: " .. prediction)

        print("Was the answer correct? (yes/no)")
        local feedback = io.read()

        if feedback == "no" then
            print("What is the correct answer? (yes/no)")
            local correct_answer = io.read()
            local correct_label = correct_answer == "yes" and 1 or 0
            table.insert(training_data, {question = question, answer = correct_label})
            table.insert(inputs, question_to_tensor(question, vocabulary, input_size))
            table.insert(labels, {correct_label})

            -- Train on the new data point
            net:train(inputs[#inputs], labels[#labels], learning_rate)
        end
    end
end

-- Start the user interaction loop
ask_questions()
